use crate::generate::heightmap::HeightmapCache;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::LayerType;
use anyhow::Error;
use std::sync::Arc;
use std::thread;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

#[derive(Debug)]
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
    receiver: UnboundedReceiver<TileResult>,
    num_inflight: usize,
}
impl TileStreamerEndpoint {
    pub(crate) fn new(mapfile: Arc<MapFile>) -> Result<Self, Error> {
        let (sender, requests) = unbounded_channel();
        let (results, receiver) = unbounded_channel();

        let mut rt = Runtime::new()?;
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
    results: UnboundedSender<TileResult>,
    mapfile: Arc<MapFile>,
    heightmap_tiles: HeightmapCache,
}

impl TileStreamer {
    async fn run(mut self) -> Result<(), Error> {
        while let Some(request) = self.requests.recv().await {
            let data = match request.layer {
                LayerType::Heightmaps => {
                    let heights: Vec<_> = self
                        .heightmap_tiles
                        .get_tile(&*self.mapfile, request.node)
                        .await
                        .unwrap()
                        .iter()
                        .map(|&i| i as f32)
                        .collect();
                    let mut data = vec![0; heights.len() * 4];
                    data.copy_from_slice(bytemuck::cast_slice(&heights));
                    data
                }
                _ => self.mapfile.read_tile(request.layer, request.node).unwrap(),
            };
            self.results.send(TileResult { node: request.node, layer: request.layer, data })?;
        }
        Ok(())
    }
}
