
__kernel void compute_pressure(const int num_points, const float d2, const float r2, const float mass, const float poly6kern, const float prest_densisty, const float pintstiff
	,global float* pos, global float* pressure, global float* density) {

	int i = get_global_id(0);
	int i_idx = i * 3;

	const float ipos_x = pos[i_idx+0];
	const float ipos_y = pos[i_idx+1];
	const float ipos_z = pos[i_idx+2];

	float sum = 0.0;

	for (int j = 0; j<num_points; ++j) {
		if (i==j) continue;

		int j_idx = j*3;
		float dst_x = pos[j_idx+0] - ipos_x;
		float dst_y = pos[j_idx+1] - ipos_y;
		float dst_z = pos[j_idx+2] - ipos_z;

		float dsq = d2*(dst_x*dst_x + dst_y*dst_y + dst_z*dst_z);
		if (dsq <= r2) {
			float c =  r2 - dsq;
			sum += c * c * c;
		}

		float tmp_density = sum * mass * poly6kern;
		pressure[i] = (tmp_density - prest_densisty) * pintstiff;
		density[i] = 1.0f / tmp_density;
	}

}

__kernel void compute_force(const int num_points, const float d2, const float r, const float r2, const float d5spikykern, const float vterm
	,global float* pos, global float* pressure, global float* density, global float* veleval, global float* force) {

	const int i = get_global_id(0);
	const int i_idx = i * 3;

	const float ipos_x = pos[i_idx+0];
	const float ipos_y = pos[i_idx+1];
	const float ipos_z = pos[i_idx+2];

	force[i_idx+0] = 0;
	force[i_idx+1] = 0;
	force[i_idx+2] = 0;

	const float ipress = pressure[i];
	const float idensity = density[i];
	const float iveleval_x = veleval[i_idx+0];
	const float iveleval_y = veleval[i_idx+1];
	const float iveleval_z = veleval[i_idx+2];

	for (int j = 0; j < num_points; ++j) {
		if (i==j) continue;

		const int j_idx = j*3;
		const float dx = ipos_x - pos[j_idx+0]; //dist in cm
		const float dy = ipos_y - pos[j_idx+1];
		const float dz = ipos_z - pos[j_idx+2];

		const float dsq = d2*(dx*dx + dy*dy + dz*dz);

		if (dsq <= r2) {
			const float jdist = sqrt(dsq);

			const float jpress = pressure[j];
			const float jdensity = density[j];
			const float jveleval_x = veleval[j_idx+0];
			const float jveleval_y = veleval[j_idx+1];
			const float jveleval_z = veleval[j_idx+2];

			const float c = (r - jdist);
			const float pterm = c * d5spikykern * ( ipress + jpress ) / jdist;
			const float dterm = c * idensity * jdensity;
			force[i_idx+0] += ( pterm * dx + vterm * ( jveleval_x - iveleval_x) ) * dterm;
			force[i_idx+1] += ( pterm * dy + vterm * ( jveleval_y - iveleval_y) ) * dterm;
			force[i_idx+2] += ( pterm * dz + vterm * ( jveleval_z - iveleval_z) ) * dterm;
		}
	}

}
