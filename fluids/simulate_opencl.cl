
__kernel void compute_pressure(const int num_points, const float d2, const float r2, const float mass, const float poly6kern, const float prest_densisty, const float pintstiff
	,global float* pos, global float* pressure, global float* density) {

	const int i = get_global_id(0);
	const int i_idx = i * 3;

	const float3 ipos = (float3) (pos[i_idx+0], pos[i_idx+1], pos[i_idx+2]);

	float sum = 0.0;

	for (int j = 0; j<num_points; ++j) {
		if (i==j) continue;

		const int j_idx = j*3;
		const float3 dst = ((float3) (pos[j_idx+0], pos[j_idx+1], pos[j_idx+2])) - ipos;

		float dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
		if (dsq <= r2) {
			const float c =  r2 - dsq;
			sum += c * c * c;
		}

		const float tmp_density = sum * mass * poly6kern;
		pressure[i] = (tmp_density - prest_densisty) * pintstiff;
		density[i] = 1.0f / tmp_density;
	}

}

__kernel void compute_force(const int num_points, const float d2, const float r, const float r2, const float d5spikykern, const float vterm
	,global float* pos, global float* pressure, global float* density, global float* veleval, global float3* force) {

	const int i = get_global_id(0);
	const int i_idx = i * 3;

	force[i] = (float3) (0, 0, 0);

	const float ipress = pressure[i];
	const float idensity = density[i];
	const float3 ipos = (float3) (pos[i_idx+0], pos[i_idx+1], pos[i_idx+2]);
	const float3 iveleval = (float3) (veleval[i_idx+0], veleval[i_idx+1], veleval[i_idx+2]);

	for (int j = 0; j < num_points; ++j) {
		if (i==j) continue;

		const int j_idx = j*3;
		const float3 dst = ipos - ((float3) (pos[j_idx+0], pos[j_idx+1], pos[j_idx+2]));

		const float dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);;

		if (dsq <= r2) {
			const float jdist = sqrt(dsq);

			const float jpress = pressure[j];
			const float jdensity = density[j];
			const float3 jveleval = (float3) (veleval[j_idx+0], veleval[j_idx+1], veleval[j_idx+2]);

			const float c = (r - jdist);
			const float pterm = c * d5spikykern * ( ipress + jpress ) / jdist;
			const float dterm = c * idensity * jdensity;
			force[i].x += ( pterm * dst.x + vterm * ( jveleval.x - iveleval.x) ) * dterm;
			force[i].y += ( pterm * dst.y + vterm * ( jveleval.y - iveleval.y) ) * dterm;
			force[i].z += ( pterm * dst.z + vterm * ( jveleval.z - iveleval.z) ) * dterm;
		}
	}

}
