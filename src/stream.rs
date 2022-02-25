use crate::cache::LayerType;
use crate::generate::heightmap::HeightmapCache;
use crate::mapfile::MapFile;
use anyhow::Error;
use futures::{FutureExt, StreamExt};
use std::sync::Arc;
use std::thread;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use types::VNode;

#[derive(Copy, Clone, Debug)]
struct TileRequest {
    node: VNode,
    layer: LayerType,
}

#[derive(Debug)]
pub(crate) enum TileResult {
    Heightmaps(VNode, Arc<Vec<i16>>),
    Generic(VNode, LayerType, Vec<u8>),
}
impl TileResult {
    pub fn layer(&self) -> LayerType {
        match self {
            TileResult::Heightmaps(..) => LayerType::Heightmaps,
            TileResult::Generic(_, ty, _) => *ty,
        }
    }
    pub fn node(&self) -> VNode {
        match self {
            TileResult::Heightmaps(node, ..) | TileResult::Generic(node, ..) => *node,
        }
    }
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
            rt.block_on(
                TileStreamer {
                    requests,
                    results,
                    heightmap_tiles: HeightmapCache::new(
                        mapfile.layers()[LayerType::Heightmaps].texture_resolution as usize,
                        mapfile.layers()[LayerType::Heightmaps].texture_border_size as usize,
                        128,
                    ),
                    mapfile,
                }
                .run(),
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

                    pending.push(match request.layer {
                        LayerType::Heightmaps => {
                            let fut = heightmap_tiles.get_tile(mapfile, request.node);
                            async move {
                                Ok(TileResult::Heightmaps(request.node, fut.await?))
                            }.boxed()
                        }
                        _ => async move {
                            let data = match mapfile.read_tile(request.layer, request.node).await? {
                                Some(raw_data) => {
                                    tokio::task::spawn_blocking(move || {
                                        let img = image::load_from_memory(&raw_data)?;
                                        Ok::<Vec<u8>, Error>(match request.layer {
                                            LayerType::BaseAlbedo => img.to_rgba8().to_vec(),
                                            LayerType::TreeCover => img.to_luma8().to_vec(),
                                            LayerType::WaterMask => img.to_luma8().to_vec(),
                                            _ => unreachable!(),
                                        })
                                    }).await??
                                }
                                None => Vec::new(),
                            };
                            Ok::<TileResult, Error>(TileResult::Generic(request.node, request.layer, data))
                        }.boxed()
                    });
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
