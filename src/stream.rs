use crate::cache::layer::LayerType;
use crate::mapfile::MapFile;
use anyhow::Error;
use futures::{FutureExt, StreamExt};
use std::io::{Cursor, Read};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use types::VNode;
use vec_map::VecMap;
use zip::result::ZipError;

#[derive(Debug)]
pub(crate) struct TileResult {
    pub node: VNode,
    pub layers: VecMap<Vec<u8>>,
}

pub(crate) struct TileStreamerEndpoint {
    sender: UnboundedSender<(VNode, Instant)>,
    receiver: crossbeam::channel::Receiver<TileResult>,
    join_handle: Option<thread::JoinHandle<()>>,
    num_inflight: usize,
}
impl TileStreamerEndpoint {
    pub(crate) fn new(
        mapfile: Arc<MapFile>,
        transcode_format: wgpu::TextureFormat,
    ) -> Result<Self, Error> {
        let (sender, requests) = unbounded_channel();
        let (results, receiver) = crossbeam::channel::unbounded();

        let rt = Runtime::new()?;
        let join_handle = Some(thread::spawn(move || {
            rt.block_on(
                TileStreamer {
                    requests,
                    results,
                    // heightmap_tiles: HeightmapCache::new(
                    //     mapfile.layers()[LayerType::Heightmaps].texture_resolution as usize,
                    //     mapfile.layers()[LayerType::Heightmaps].texture_border_size as usize,
                    //     128,
                    // ),
                    transcode_format,
                    mapfile,
                }
                .run(),
            )
            .unwrap();
        }));

        Ok(Self { sender, receiver, join_handle, num_inflight: 0 })
    }

    pub(crate) fn request_tile(&mut self, node: VNode) {
        if let Err(_) = self.sender.send((node, Instant::now())) {
            // The worker thread has panicked (we still have the sender open, so that cannot be why
            // it exited).
            self.join_handle.take().unwrap().join().unwrap();
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
    requests: UnboundedReceiver<(VNode, Instant)>,
    results: crossbeam::channel::Sender<TileResult>,
    transcode_format: wgpu::TextureFormat,
    mapfile: Arc<MapFile>,
}

impl TileStreamer {
    fn parse_tile(
        node: VNode,
        bytes: &[u8],
        _transcode_format: wgpu::TextureFormat,
    ) -> Result<TileResult, Error> {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes))?;
        let mut result = TileResult { node, layers: VecMap::new() };

        let mut get_file = |name| -> Result<Option<Vec<u8>>, Error> {
            match zip.by_name(name) {
                Ok(mut file) => {
                    let mut bytes = Vec::new();
                    file.read_to_end(&mut bytes)?;
                    Ok(Some(bytes))
                }
                Err(ZipError::FileNotFound) => Ok(None),
                Err(e) => Err(e.into()),
            }
        };

        let decode_nonempty = |bytes: Vec<u8>| -> Result<Option<Vec<u8>>, Error> {
            if bytes.is_empty() {
                Ok(None)
            } else {
                Ok(Some(zstd::decode_all(Cursor::new(
                    &ktx2::Reader::new(bytes)?.levels().next().expect("ktx2 has no levels"),
                ))?))
            }
        };

        result.layers.insert(
            LayerType::BaseHeightmaps.index(),
            decode_nonempty(get_file("heights.ktx2")?.expect("layer missing"))?
                .unwrap_or_else(|| vec![0u8; 521 * 521 * 2]),
        );
        result.layers.insert(
            LayerType::TreeCover.index(),
            decode_nonempty(get_file("treecover.ktx2")?.expect("layer missing"))?
                .unwrap_or_else(|| vec![0u8; 516 * 516]),
        );
        result.layers.insert(
            LayerType::LandFraction.index(),
            decode_nonempty(get_file("landfraction.ktx2")?.expect("layer missing"))?
                .unwrap_or_else(|| vec![0u8; 516 * 516]),
        );

        if let Some(bytes) = get_file("waterlevel.ktx2")? {
            result.layers.insert(
                LayerType::WaterLevel.index(),
                decode_nonempty(bytes)?.unwrap_or_else(|| vec![0u8; 521 * 521 * 2]),
            );
        }
        if let Some(bytes) = get_file("albedo.ktx2")? {
            result.layers.insert(
                LayerType::BaseAlbedo.index(),
                decode_nonempty(bytes)?.unwrap_or_else(|| vec![0u8; 516 * 516 * 4]),
            );
        }

        if node.level() == 0 {
            assert!(result.layers.contains_key(LayerType::BaseHeightmaps.index()));
            assert!(result.layers.contains_key(LayerType::TreeCover.index()));
            assert!(result.layers.contains_key(LayerType::LandFraction.index()));
            assert!(result.layers.contains_key(LayerType::BaseAlbedo.index()));
        }

        Ok(result)
    }

    async fn run(self) -> Result<(), Error> {
        let TileStreamer { mut requests, results, mapfile, transcode_format } = self;
        let mapfile = &*mapfile;

        let mut pending = futures::stream::futures_unordered::FuturesUnordered::new();
        loop {
            futures::select! {
                tile_result = pending.select_next_some() => {
                    results.send(tile_result?)?;
                },
                node = requests.recv().fuse() => if let Some((node, _start)) = node {
                    pending.push(async move {
                        match mapfile.read_tile(node).await? {
                            Some(raw_data) => {
                                tokio::task::spawn_blocking(move || Self::parse_tile(node, &raw_data, transcode_format)).await.unwrap()
                            }
                            None => {
                                let mut result = TileResult {
                                    node,
                                    layers: VecMap::new(),
                                };
                                result.layers.insert(LayerType::BaseHeightmaps.index(), bytemuck::cast_slice(&vec![0u16; 521 * 521]).to_vec());
                                result.layers.insert(LayerType::TreeCover.index(), vec![0u8; 516 * 516]);
                                result.layers.insert(LayerType::LandFraction.index(), vec![0u8; 516 * 516]);
                                Ok(result)
                            }
                        }
                    }.boxed());
                },
                complete => break,
            }
        }
        Ok(())
    }
}
