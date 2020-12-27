use crate::generate::heightmap::HeightmapCache;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::LayerType;
use anyhow::Error;
use futures::{FutureExt, StreamExt};
use std::io::{Cursor, Read};
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
    join_handle: Option<thread::JoinHandle<Result<(), Error>>>,
    num_inflight: usize,
}
impl TileStreamerEndpoint {
    pub(crate) fn new(mapfile: Arc<MapFile>) -> Result<Self, Error> {
        let (sender, requests) = unbounded_channel();
        let (results, receiver) = crossbeam::channel::unbounded();

        let rt = Runtime::new()?;
        let join_handle = Some(thread::spawn(move || {
            rt.block_on(TileStreamer {
                    requests,
                    results,
                    heightmap_tiles: HeightmapCache::new(
                        mapfile.layers()[LayerType::Heightmaps].clone(),
                        32,
                    ),
                    mapfile,
                }
                .run()
            )
        }));

        Ok(Self { sender, receiver, join_handle, num_inflight: 0 })
    }

    pub(crate) fn request_tile(&mut self, node: VNode, layer: LayerType) {
        if let Err(_) = self.sender.send(TileRequest { node, layer }) {
            // The worker thread has panicked (we still have the sender open, so that cannot be why 
            // it exited). Join it to see what the panic message was.
            self.join_handle.take().unwrap().join().unwrap().expect("TileStreamer panicked");
            unreachable!("TileStreamer exited without panicking");
        }
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
    async fn run(self) -> Result<(), Error> {
        let TileStreamer { mut requests, results, mapfile, mut heightmap_tiles } = self;
        let mapfile = &*mapfile;

        let mut pending = futures::stream::futures_unordered::FuturesUnordered::new();
        loop {
            futures::select! {
                request = requests.recv().fuse() => if let Some(request) = request {
                    match request.layer {
                        LayerType::Heightmaps => {
                            let fut = heightmap_tiles.get_tile(mapfile, request.node);

                            pending.push(async move {
                                let heights: Vec<_> = fut
                                    .await
                                    .unwrap()
                                    .iter()
                                    .map(|&i| i as f32)
                                    .collect();
                                let mut data = vec![0; heights.len() * 4];
                                data.copy_from_slice(bytemuck::cast_slice(&heights));
                                Ok(TileResult { node: request.node, layer: request.layer, data })
                            }.boxed());
                        }
                        LayerType::Albedo => pending.push(async move {
                            let raw_data = mapfile.read_tile(request.layer, request.node).await?;
                            let data = tokio::task::spawn_blocking(move || {
                                Ok::<Vec<u8>, Error>(image::load_from_memory(&raw_data)?.to_rgba8().to_vec())
                            }).await??;
                            Ok::<TileResult, Error>(TileResult { node: request.node, layer: request.layer, data })
                        }.boxed()),
                        LayerType::Roughness => pending.push(async move {
                            let mut data = Vec::new();
                            let raw_data = mapfile.read_tile(request.layer, request.node).await?;
                            lz4::Decoder::new(Cursor::new(&raw_data))?.read_to_end(&mut data)?;
                            Ok::<TileResult, Error>(TileResult { node: request.node, layer: request.layer, data })
                        }.boxed()),
                        LayerType::Normals | LayerType::Displacements => unreachable!(),
                    }
                },
                tile_result = pending.select_next_some() => {
                    results.send(tile_result?)?;
                },
                complete => break,
            }
        }
        Ok(())
    }
}
