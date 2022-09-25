use crate::cache::LayerType;
// use crate::generate::heightmap::HeightmapCache;
use crate::mapfile::MapFile;
use anyhow::Error;
use basis_universal::{Transcoder, TranscoderTextureFormat};
use futures::{FutureExt, StreamExt};
use std::io::{Cursor, Read};
use std::sync::Arc;
use std::thread;
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
    sender: UnboundedSender<VNode>,
    receiver: crossbeam::channel::Receiver<TileResult>,
    join_handle: Option<thread::JoinHandle<Result<(), Error>>>,
    num_inflight: usize,
}
impl TileStreamerEndpoint {
    pub(crate) fn new(
        mapfile: Arc<MapFile>,
        transcode_format: TranscoderTextureFormat,
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
        if let Err(_) = self.sender.send(node) {
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
    requests: UnboundedReceiver<VNode>,
    results: crossbeam::channel::Sender<TileResult>,
    transcode_format: TranscoderTextureFormat,
    mapfile: Arc<MapFile>,
}

impl TileStreamer {
    fn parse_tile(
        node: VNode,
        bytes: &[u8],
        transcode_format: TranscoderTextureFormat,
    ) -> Result<TileResult, Error> {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes))?;
        let mut result =
            TileResult { node, heightmap: vec![0i16; 517 * 517], layers: VecMap::new() };

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

        if let Some(bytes) = get_file("albedo.basis")? {
            let mut transcoder = Transcoder::new();
            transcoder.prepare_transcoding(&bytes).unwrap();
            let data = transcoder
                .transcode_image_level(&bytes, transcode_format, Default::default())
                .map_err(|e| anyhow::format_err!("corrupt albedo.basis: {:?}", e))?;
            transcoder.end_transcoding();
            result.layers.insert(LayerType::BaseAlbedo.index(), data);
        }

        let heights: Vec<_> = result
            .heightmap
            .iter()
            .map(|&h| {
                if h <= 0 {
                    0x800000 | (((h + 1024).max(0) as u32) << 9)
                } else {
                    (((h as u32) + 1024) << 9).min(0x7fffff)
                }
            })
            .collect();
        result
            .layers
            .insert(LayerType::Heightmaps.index(), bytemuck::cast_slice(&heights).to_vec());

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
                node = requests.recv().fuse() => if let Some(node) = node {
                    pending.push(async move {
                        match mapfile.read_tile(node).await? {
                            Some(raw_data) => {
                                tokio::task::spawn_blocking(move || Self::parse_tile(node, &raw_data, transcode_format)).await?
                            }
                            None => {
                                let mut result = TileResult {
                                    node,
                                    heightmap: vec![0i16; 517 * 517],
                                    layers: VecMap::new(),
                                };
                                result.layers.insert(LayerType::TreeCover.index(), vec![0u8; 516 * 516]);
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
