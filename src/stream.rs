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
    pub heightmap: Vec<i16>,
    pub layers: VecMap<Vec<u8>>,
}

pub(crate) struct TileStreamerEndpoint {
    sender: UnboundedSender<(VNode, Instant)>,
    receiver: crossbeam::channel::Receiver<TileResult>,
    join_handle: Option<thread::JoinHandle<Result<(), Error>>>,
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
        }));

        Ok(Self { sender, receiver, join_handle, num_inflight: 0 })
    }

    pub(crate) fn request_tile(&mut self, node: VNode) {
        if let Err(_) = self.sender.send((node, Instant::now())) {
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
        let mut result =
            TileResult { node, heightmap: vec![0i16; 521 * 521], layers: VecMap::new() };

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

        if let Some(compressed) = get_file("heights.lz4")? {
            if !compressed.is_empty() {
                lz4::Decoder::new(Cursor::new(compressed))?
                    .read_exact(bytemuck::cast_slice_mut(&mut result.heightmap))?;
                let mut prev = 0;
                for i in 0..result.heightmap.len() {
                    result.heightmap[i] = result.heightmap[i].wrapping_add(prev);
                    prev = result.heightmap[i];
                }
            }
        }

        if let Some(compressed) = get_file("waterlevel.lz4")? {
            let mut waterlevel = vec![0i16; 521 * 521];
            if !compressed.is_empty() {
                lz4::Decoder::new(Cursor::new(compressed))?
                    .read_exact(bytemuck::cast_slice_mut(&mut waterlevel))?;
                let mut prev = 0;
                for i in 0..waterlevel.len() {
                    waterlevel[i] = waterlevel[i].wrapping_add(prev);
                    prev = waterlevel[i];
                }
            }
            let waterlevel: Vec<_> = waterlevel.iter().map(|&h| (h + 4096).max(0) as u16).collect();
            result.layers.insert(LayerType::WaterLevel.index(), bytemuck::cast_slice(&waterlevel).to_vec());
        }

        let mut treecover = vec![0u8; 516 * 516];
        if let Some(compressed) = get_file("treecover.lz4")? {
            if !compressed.is_empty() {
                lz4::Decoder::new(Cursor::new(compressed))?
                    .read_exact(bytemuck::cast_slice_mut(&mut treecover))?;
                let mut prev = 0;
                for i in 0..treecover.len() {
                    treecover[i] = treecover[i].wrapping_add(prev);
                    prev = treecover[i];
                }
            }
        }
        result.layers.insert(LayerType::TreeCover.index(), treecover);

        let mut landfraction = vec![0u8; 516 * 516];
        if let Some(compressed) = get_file("landfraction.lz4")? {
            if !compressed.is_empty() {
                lz4::Decoder::new(Cursor::new(compressed))?
                    .read_exact(bytemuck::cast_slice_mut(&mut landfraction))?;
                let mut prev = 0;
                for i in 0..landfraction.len() {
                    landfraction[i] = landfraction[i].wrapping_add(prev);
                    prev = landfraction[i];
                }
            }
        } else {
            assert!(node.level() > 0);
        }
        result.layers.insert(LayerType::LandFraction.index(), landfraction);

        if let Some(bytes) = get_file("albedo.ktx2")? {
            if !bytes.is_empty() {
                result.layers.insert(
                    LayerType::BaseAlbedo.index(),
                    ktx2::Reader::new(bytes)?.levels().next().unwrap().to_vec(),
                );
            } else {
                result.layers.insert(LayerType::BaseAlbedo.index(), vec![0u8; 516 * 516 * 4]);
            }
        }

        let heights: Vec<_> = result.heightmap.iter().map(|&h| (h + 4096).max(0) as u16).collect();
        result
            .layers
            .insert(LayerType::BaseHeightmaps.index(), bytemuck::cast_slice(&heights).to_vec());

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
                                    heightmap: vec![0i16; 521 * 521],
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
