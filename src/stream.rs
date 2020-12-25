use crate::generate::heightmap::HeightmapCache;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::LayerType;
use anyhow::Error;
use futures::{FutureExt, StreamExt};
use std::sync::Arc;
use std::thread;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

#[derive(Copy, Clone, Debug)]
struct TileRequest {
    node: VNode,
    layer: LayerType,
}

#[derive(Debug)]
pub(crate) struct TileResult {
    pub node: VNode,
    pub layer: LayerType,
    pub data: Vec<u8>,
}

pub(crate) struct TileStreamerEndpoint {
    sender: UnboundedSender<TileRequest>,
    receiver: crossbeam::channel::Receiver<TileResult>,
    num_inflight: usize,
}
impl TileStreamerEndpoint {
    pub(crate) fn new(mapfile: Arc<MapFile>) -> Result<Self, Error> {
        let (sender, requests) = unbounded_channel();
        let (results, receiver) = crossbeam::channel::unbounded();

        let rt = Runtime::new()?;
        thread::spawn(move || {
            rt.block_on(async {
                TileStreamer {
                    requests,
                    results,
                    heightmap_tiles: HeightmapCache::new(
                        mapfile.layers()[LayerType::Heightmaps].clone(),
                        32,
                    ),
                    mapfile,
                }
                .run()
                .await
            })
        });

        Ok(Self { sender, receiver, num_inflight: 0 })
    }

    pub(crate) fn request_tile(&mut self, node: VNode, layer: LayerType) {
        self.sender.send(TileRequest { node, layer }).unwrap();
        self.num_inflight += 1;
    }

    pub(crate) fn try_complete(&mut self) -> Option<TileResult> {
        if let Ok(result) = self.receiver.try_recv() {
            self.num_inflight -= 1;
            Some(result)
        } else {
            None
        }
    }

    pub(crate) fn num_inflight(&self) -> usize {
        self.num_inflight
    }
}

struct TileStreamer {
    requests: UnboundedReceiver<TileRequest>,
    results: crossbeam::channel::Sender<TileResult>,
    mapfile: Arc<MapFile>,
    heightmap_tiles: HeightmapCache,
}

impl TileStreamer {
    async fn run(mut self) -> Result<(), Error> {
        let mut pending = futures::stream::futures_unordered::FuturesUnordered::new();

        loop {
            futures::select! {
                request = self.requests.recv().fuse() => if let Some(request) = request {
                    match request.layer {
                        LayerType::Heightmaps => {
                            let fut = self
                                .heightmap_tiles
                                .get_tile(&*self.mapfile, request.node);

                            pending.push(async move {
                                let heights: Vec<_> = fut
                                    .await
                                    .unwrap()
                                    .iter()
                                    .map(|&i| i as f32)
                                    .collect();
                                let mut data = vec![0; heights.len() * 4];
                                data.copy_from_slice(bytemuck::cast_slice(&heights));
                                TileResult { node: request.node, layer: request.layer, data }
                            });
                        }
                        _ => self.results.send(TileResult { node: request.node, layer: request.layer, data: self.mapfile.read_tile(request.layer, request.node).unwrap() })?,
                    }
                },
                tile_result = pending.select_next_some() => {
                    self.results.send(tile_result)?;
                },
                complete => break,
            }
        }
        Ok(())
    }
}
