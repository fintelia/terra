use super::*;
use crate::cache::{CacheLookup, LayerType, SingularLayerType, UnifiedPriorityCache};
use std::mem;

#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub(crate) struct NodeState {
    displacements_desc: [[f32; 4]; 2],
    albedo_desc: [[f32; 4]; 2],
    roughness_desc: [[f32; 4]; 2],
    normals_desc: [[f32; 4]; 2],
    grass_canopy_desc: [f32; 4],
    resolution: u32,
    face: u32,
    level: u32,
    _padding0: u32,
    relative_position: [f32; 3],
    min_distance: f32,
    parent_relative_position: [f32; 3],
    _padding1: [u32; 17],
    // side_length: f32,
    // padding0: f32,
    // padding1: u32,
}
unsafe impl bytemuck::Pod for NodeState {}
unsafe impl bytemuck::Zeroable for NodeState {}

impl QuadTree {
    pub fn find_descs(
        node: VNode,
        cache: &UnifiedPriorityCache,
        ty: LayerType,
        texture_origin: Vector2<f32>,
        base_origin: Vector2<f32>,
        texture_ratio: f32,
        texture_step: f32,
    ) -> ([[f32; 4]; 2], VNode) {
        if cache.tiles.contains(node, ty) {
            let child_slot = cache.tiles.get_slot(node).expect("child_slot") as f32;
            let child_offset = texture_origin + texture_ratio * base_origin;

            if let Some((parent, child_index)) = node.parent() {
                if cache.tiles.contains(parent, ty) {
                    let parent_slot = cache.tiles.get_slot(parent).unwrap() as f32;
                    let parent_offset = node::OFFSETS[child_index as usize].cast().unwrap();
                    let parent_offset =
                        texture_origin + 0.5 * texture_ratio * (base_origin + parent_offset);

                    return (
                        [
                            [child_offset.x, child_offset.y, child_slot, texture_step],
                            [parent_offset.x, parent_offset.y, parent_slot, texture_step * 0.5],
                        ],
                        node,
                    );
                }
            }

            (
                [[child_offset.x, child_offset.y, child_slot, texture_step], [0.0, 0.0, -1.0, 0.0]],
                node,
            )
        } else {
            let (ancestor, generations, offset) = node
                .find_ancestor(|n| cache.tiles.contains(n, ty))
                .expect(&format!("find_ancestor({:?})", ty));
            let slot = cache.tiles.get_slot(ancestor).map(|s| s as f32).expect("slot");
            let scale = (0.5f32).powi(generations as i32);
            let offset = Vector2::new(offset.x as f32, offset.y as f32);
            let offset = texture_origin + scale * texture_ratio * (base_origin + offset);

            ([[offset.x, offset.y, slot, scale * texture_step], [0.0, 0.0, -1.0, 0.0]], ancestor)
        }
    }

    fn lookup_to_desc(
        lookup: CacheLookup,
        texture_origin: Vector2<f32>,
        base_origin: Vector2<f32>,
        texture_ratio: f32,
        texture_step: f32,
    ) -> [f32; 4] {
        let scale = (0.5f32).powi(lookup.levels as i32);
        let offset = Vector2::new(lookup.offset.x as f32, lookup.offset.y as f32);
        let offset = texture_origin + scale * texture_ratio * (base_origin + offset);

        [offset.x, offset.y, lookup.slot as f32, scale * texture_step]
    }

    pub fn prepare_vertex_buffer(
        &mut self,
        queue: &wgpu::Queue,
        vertex_buffer: &wgpu::Buffer,
        cache: &UnifiedPriorityCache,
        camera: mint::Point3<f64>,
    ) {
        assert_eq!(
            cache.tile_desc(LayerType::Albedo).texture_resolution,
            cache.tile_desc(LayerType::Normals).texture_resolution
        );
        assert_eq!(
            cache.tile_desc(LayerType::Albedo).texture_border_size,
            cache.tile_desc(LayerType::Normals).texture_border_size
        );

        let resolution = cache.tile_desc(LayerType::Displacements).texture_resolution - 1;
        let texture_resolution = cache.tile_desc(LayerType::Normals).texture_resolution;
        let texture_border = cache.tile_desc(LayerType::Normals).texture_border_size;
        let texture_ratio =
            (texture_resolution - 2 * texture_border) as f32 / texture_resolution as f32;
        let texture_step = texture_ratio / resolution as f32;
        let texture_origin = texture_border as f32 / texture_resolution as f32;

        self.node_states.clear();
        for &node in self.visible_nodes.iter() {
            assert!(node.min_distance() as f32 != 0.0);
            let (displacements_desc, displacements_node) = Self::find_descs(
                node,
                &cache,
                LayerType::Displacements,
                Vector2::new(0.5, 0.5) / (resolution + 1) as f32,
                Vector2::new(0.0, 0.0),
                resolution as f32 / (resolution + 1) as f32,
                1.0 / (resolution + 1) as f32,
            );
            let albedo_desc = Self::find_descs(
                node,
                &cache,
                LayerType::Albedo,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            )
            .0;
            let roughness_desc = Self::find_descs(
                node,
                &cache,
                LayerType::Roughness,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            )
            .0;
            let normals_desc = Self::find_descs(
                node,
                &cache,
                LayerType::Normals,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            )
            .0;
            let grass_canopy_desc = cache.lookup_texture(SingularLayerType::GrassCanopy, node).map(|lookup| Self::lookup_to_desc(
                lookup,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            )).unwrap_or([0.0, 0.0, -1.0, 0.0]);
            self.node_states.push(NodeState {
                _padding0: 0,
                _padding1: [0; 17],
                min_distance: node.min_distance() as f32,
                displacements_desc,
                albedo_desc,
                roughness_desc,
                normals_desc,
                grass_canopy_desc,
                resolution,
                face: node.face() as u32,
                level: node.level() as u32,
                relative_position: (cgmath::Point3::from(camera)
                    - displacements_node.center_wspace())
                .cast::<f32>()
                .unwrap()
                .into(),
                parent_relative_position: (cgmath::Point3::from(camera)
                    - displacements_node.parent().map(|x| x.0).unwrap_or(node).center_wspace())
                .cast::<f32>()
                .unwrap()
                .into(),
            });
        }
        for &(node, mask) in self.partially_visible_nodes.iter() {
            assert!(mask < 15);
            assert!(node.min_distance() as f32 != 0.0);
            for i in 0..4u8 {
                if mask & (1 << i) != 0 {
                    let offset = ((i % 2) as f32, (i / 2) as f32);
                    let base_origin = Vector2::new(offset.0 * (0.5), offset.1 * (0.5));
                    let (displacements_desc, displacements_node) = Self::find_descs(
                        node,
                        &cache,
                        LayerType::Displacements,
                        Vector2::new(0.5, 0.5) / (resolution + 1) as f32,
                        Vector2::new(offset.0, offset.1) * 0.5,
                        resolution as f32 / (resolution + 1) as f32,
                        1.0 / (resolution + 1) as f32,
                    );
                    let albedo_desc = Self::find_descs(
                        node,
                        &cache,
                        LayerType::Albedo,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    )
                    .0;
                    let roughness_desc = Self::find_descs(
                        node,
                        &cache,
                        LayerType::Roughness,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    )
                    .0;
                    let normals_desc = Self::find_descs(
                        node,
                        &cache,
                        LayerType::Normals,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    )
                    .0;
                    let grass_canopy_desc = cache.lookup_texture(SingularLayerType::GrassCanopy, node).map(|lookup| Self::lookup_to_desc(
                        lookup,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    )).unwrap_or([0.0, 0.0, -1.0, 0.0]);
                    self.node_states.push(NodeState {
                        _padding0: 0,
                        _padding1: [0; 17],
                        // side_length: node.side_length() * 0.5,
                        min_distance: node.min_distance() as f32,
                        displacements_desc,
                        albedo_desc,
                        roughness_desc,
                        normals_desc,
                        grass_canopy_desc,
                        resolution: resolution / 2,
                        face: node.face() as u32,
                        level: node.level() as u32,
                        relative_position: (cgmath::Point3::from(camera)
                            - displacements_node.center_wspace())
                        .cast::<f32>()
                        .unwrap()
                        .into(),
                        parent_relative_position: (cgmath::Point3::from(camera)
                            - displacements_node
                                .parent()
                                .map(|x| x.0)
                                .unwrap_or(node)
                                .center_wspace())
                        .cast::<f32>()
                        .unwrap()
                        .into(),
                    });
                }
            }
        }

        assert_eq!(mem::size_of::<NodeState>(), 256);
        queue.write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&self.node_states));
    }

    pub(crate) fn render<'b, 'c>(
        &self,
        rpass: &'b mut wgpu::RenderPass<'c>,
        index_buffer: &'c wgpu::Buffer,
        bind_group: &'c wgpu::BindGroup,
    ) {
        let resolution = self.heights_resolution;
        let visible_nodes = self.visible_nodes.len() as u32;
        let total_nodes = self.node_states.len() as u32;

        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        for i in 0..visible_nodes {
            rpass.set_bind_group(0, &bind_group, &[mem::size_of::<NodeState>() as u32 * i]);
            rpass.draw_indexed(0..(resolution * resolution * 6), 0, 0..1);
        }

        for i in visible_nodes..total_nodes {
            rpass.set_bind_group(0, &bind_group, &[mem::size_of::<NodeState>() as u32 * i]);
            rpass.draw_indexed(
                (resolution * resolution * 6)
                    ..((resolution * resolution * 6) + ((resolution / 2) * (resolution / 2) * 6)),
                0,
                0..1,
            );
        }
    }
}
