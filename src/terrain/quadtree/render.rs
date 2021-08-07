use super::*;

impl QuadTree {
    pub(crate) fn render<'b, 'c>(
        &self,
        rpass: &'b mut wgpu::RenderPass<'c>,
        index_buffer: &'c wgpu::Buffer,
        bind_group: &'c wgpu::BindGroup,
        tile_cache: &TileCache,
    ) {
        let resolution = self.heights_resolution;
        let visible_nodes = self.visible_nodes.len() as u32;

        let num_indices_full = resolution * resolution * 6;
        let num_indices_partial = (resolution / 2) * (resolution / 2) * 6;

        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rpass.set_bind_group(0, bind_group, &[]);

        for &n in &self.visible_nodes {
            let slot = tile_cache.get_slot(n).unwrap() as u32;
            rpass.draw_indexed(0..num_indices_full, 0, slot..(slot+1));
        }

        for &(n, mask) in &self.partially_visible_nodes {
            let slot = tile_cache.get_slot(n).unwrap() as u32;
            for j in 0..4 {
                if (1<<j) & mask != 0 {
                    rpass.draw_indexed((num_indices_partial*j)..(num_indices_partial*(j+1)), 0, slot..(slot+1));
                }
            }
        }
    }
}
