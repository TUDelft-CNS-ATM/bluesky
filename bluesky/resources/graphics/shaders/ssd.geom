#version 330
#define R_EARTH 6378000
#define NM2M 1853.184
#define LOOKAHEAD_RANGE 2e5 // 200 km
#define RPZ 9265 // 5 NM
#define SSD_ALT_RANGE 304.8 // 1,000 ft
#define VSCALE 0.4 // Display scale for SSD [ms-1 / GLXY]
layout (points) in;
layout (triangle_strip, max_vertices=13) out;

in GSData {
    vec2 vAR;
    vec4 ownship;
    vec4 intruder;
	vec2 asasreso;
    float dH;
    int own_id;
    int int_id;
    int selssd;
} gs_in[];

out vec2 ssd_coord;
out vec4 color_fs;

// Vlimits is [Vmin^2, Vmax^2, Vmax]
uniform vec3 Vlimits;
uniform int n_ac;

void intersect_vmax_box(in vec2 Vint, in vec2 n, out vec2 vertex, out int segment)
{
    // We look at line-line intersections, where line 1 is Vint + n * s, and line 2 is a + b * t,
    // where a is a corner of the box, and b the directional vector of an edge
    // If an intersection exists, the value for s at the intersection is found as:
    // s = (det([a b]) - det([Vint b])) / det([n b])
    // The lines are parallel if det([n b]) is zero
    segment = -1;
    float s = 0.0, s_tmp = 0.0;
    if (abs(n.y) > 1e-6) {
        // Segment 0: top edge:    a=[-Vmax, Vmax]  b=[1, 0] ==> den = -n.y
        s_tmp = (Vlimits[2] - Vint.y) / n.y;
        if (s_tmp > s && abs(Vint.x + n.x * s_tmp) <= Vlimits[2]) {
            segment = 0;
            s = s_tmp;
        }

        // Segment 2: bottom edge: a=[-Vmax, -Vmax] b=[1, 0] ==> den = -n.y
        s_tmp = (-Vlimits[2] - Vint.y) / n.y;
        if (s_tmp > s && abs(Vint.x + n.x * s_tmp) <= Vlimits[2]) {
            segment = 2;
            s = s_tmp;
        }
    }

    if (abs(n.x) > 1e-6) {
        // Segment 1: right edge:  a=[Vmax, -Vmax]  b=[0, 1] ==> den = n.x
        s_tmp = (Vlimits[2] - Vint.x) / n.x;
        if (s_tmp > s && abs(Vint.y + n.y * s_tmp) <= Vlimits[2]) {
            segment = 1;
            s = s_tmp;
        }
    } else {
        // Segment 3: left edge:   a=[-Vmax, -Vmax] b=[0, 1] ==> den = n.x
        s_tmp = (-Vlimits[2] - Vint.x) / n.x;
        if (s_tmp > s && abs(Vint.y + n.y * s_tmp) <= Vlimits[2]) {
            segment = 3;
            s = s_tmp;
        }
    }

    // Determine the intersection vertex
    vertex = vec2(Vint + n * s);
}

void main()
{
    if (gs_in[0].selssd == 0) {
        return;
    }
    // First thing to draw is the SSD background
    if (gs_in[0].int_id == 0) {
        color_fs = vec4(0.5, 0.5, 0.5, 0.5);
        ssd_coord = Vlimits[2] * vec2(-1.0, 1.0);
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = Vlimits[2] * vec2(-1.0, -1.0);
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = Vlimits[2] * vec2(1.0, 1.0);
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = Vlimits[2] * vec2(1.0, -1.0);
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        EndPrimitive();
    }
    // VO's can be drawn for all aircraft other than ownship
    if (gs_in[0].own_id != gs_in[0].int_id) {
        // Altitude check
        if (abs(gs_in[0].dH) <= SSD_ALT_RANGE) {
            // Distance vector, flat-earth approximation
            float avglat = 0.5 * radians(gs_in[0].intruder[0] + gs_in[0].ownship[0]);
            float dlat   = radians(gs_in[0].intruder[0] - gs_in[0].ownship[0]);
            float dlon   = radians(gs_in[0].intruder[1] - gs_in[0].ownship[1]);

            vec2 dx = R_EARTH * vec2(dlon * cos(avglat), dlat);

            // Distance
            float d = length(dx);

            // Range check
            if (d <= LOOKAHEAD_RANGE && d > RPZ) {
                color_fs = vec4(1.0, 0.0, 0.0, 1.0);

                // Aircraft is within range, draw a triangular velocity obstacle
                float trkint = radians(gs_in[0].intruder[3]);
                vec2 Vint = gs_in[0].intruder[2] * vec2(sin(trkint), cos(trkint));
                vec2 nd = dx / d;

                // Rotation matrix to go from distance vector to VO legs: [r0, -r1; r1 r0]
                float r1 = RPZ / d;
                float r0 = sqrt(1.0 - r1 * r1);

                // vec2 n;
                // vec2 vertex1, vertex2;
                // int segment1, segment2;

                // // VO Leg 1
                // n = vec2(nd.x * r0 - nd.y * r1, nd.y * r0 + nd.x * r1);
                // intersect_vmax_box(Vint, n, vertex1, segment1);

                // // VO Leg 2
                // n = vec2(nd.x * r0 + nd.y * r1, nd.y * r0 - nd.x * r1);
                // intersect_vmax_box(Vint, n, vertex2, segment2);

                // if (segment1 != -1 || segment2 != -1) {
                //     // Create triangle strip
                //     if (segment1 != -1) {
                //         ssd_coord   = vertex1.xy;
                //         gl_Position = gl_in[0].gl_Position + vec4(gs_in[0].vAR * VSCALE * vertex1, 0.0, 0.0);
                //         EmitVertex();

                //         if (segment2 != segment1) {
                //             // VO legs intersect with different edges: we need to include vertices for one or two box corners
                //             gl_Position = gl_in[0].gl_Position;
                //             if (segment1 < 2) {
                //                 ssd_coord.x   = Vlimits[2];
                //                 gl_Position.x += gs_in[0].vAR[0] * VSCALE * Vlimits[2];
                //             } else {
                //                 ssd_coord.x = -Vlimits[2];
                //                 gl_Position.x -= gs_in[0].vAR[0] * VSCALE * Vlimits[2];
                //             }
                //             if (segment1 == 0 || segment1 == 3) {
                //                 ssd_coord.y = Vlimits[2];
                //                 gl_Position.y += gs_in[0].vAR[1] * VSCALE * Vlimits[2];
                //             } else {
                //                 ssd_coord.y = -Vlimits[2];
                //                 gl_Position.y -= gs_in[0].vAR[1] * VSCALE * Vlimits[2];
                //             }
                //             EmitVertex();
                //         }
                //     }

                //     // Tip of the triangle
                //     ssd_coord   = Vint;
                //     gl_Position = gl_in[0].gl_Position + vec4(gs_in[0].vAR * VSCALE * Vint, 0.0, 0.0);
                //     EmitVertex();

                //     // For very wide VO's two box corners need to be included
                //     if (abs(segment1 - segment2) > 1 || segment1 == -1) {
                //         gl_Position = gl_in[0].gl_Position;
                //         if (segment2 == 1 || segment2 == 2) {
                //             ssd_coord.x = Vlimits[2];
                //             gl_Position.x += gs_in[0].vAR[0] * VSCALE * Vlimits[2];
                //         } else {
                //             ssd_coord.x = -Vlimits[2];
                //             gl_Position.x -= gs_in[0].vAR[0] * VSCALE * Vlimits[2];
                //         }
                //         if (segment2 < 2) {
                //             ssd_coord.y = Vlimits[2];
                //             gl_Position.y += gs_in[0].vAR[1] * VSCALE * Vlimits[2];
                //         } else {
                //             ssd_coord.y = -Vlimits[2];
                //             gl_Position.y -= gs_in[0].vAR[1] * VSCALE * Vlimits[2];
                //         }
                //         EmitVertex();
                //     }

                //     // Then finally the intersection point of VO leg 2
                //     if (segment2 != -1) {
                //         ssd_coord = vertex2.xy;
                //         gl_Position = gl_in[0].gl_Position + vec4(gs_in[0].vAR * VSCALE * vertex2, 0.0, 0.0);
                //         EmitVertex();
                //     }

                // The tip of the triangle
                ssd_coord = Vint;
                gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
                EmitVertex();

                // VO Leg 1
                vec2 vertex1 = Vint + 2.0 * Vlimits[2] * vec2(nd.x * r0 - nd.y * r1, nd.y * r0 + nd.x * r1);
                ssd_coord = vertex1;
                gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
                EmitVertex();

                // VO Leg 2
                vec2 vertex2 = Vint + 2.0 * Vlimits[2] * vec2(nd.x * r0 + nd.y * r1, nd.y * r0 - nd.x * r1);
                ssd_coord = vertex2;
                gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
                EmitVertex();

                // Extension of vertex1 in the direction of the VO bisector
                ssd_coord = vertex1 + 2.0 * Vlimits[2] * nd;
                gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
                EmitVertex();

                // Extension of vertex2 in the direction of the VO bisector
                ssd_coord = vertex2 + 2.0 * Vlimits[2] * nd;
                gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
                EmitVertex();

                EndPrimitive();
            }
        }
    }
    // After the last VO draw the own velocity vector
    if (gs_in[0].int_id == n_ac - 1) {
        color_fs = vec4(0.0, 1.0, 0.0, 1.0);
        float trkown = radians(gs_in[0].ownship[3]);
        vec2 nVown = vec2(sin(trkown), cos(trkown));
        vec2 nnVown = 5.0 * vec2(nVown.y, -nVown.x);

        ssd_coord = nnVown;
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = -nnVown;
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = gs_in[0].ownship[2] * nVown + nnVown;
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        ssd_coord = gs_in[0].ownship[2] * nVown - nnVown;
        gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
        EmitVertex();
        EndPrimitive();
    }
	// After the last VO draw the asasresolution point by ASAS
	if (gs_in[0].int_id == n_ac - 1) {
		float dasasreso = gs_in[0].asasreso[0] * gs_in[0].asasreso[0] + gs_in[0].asasreso[1] * gs_in[0].asasreso[1];
		// Only draw when the asasresolution point is within the velocity limits
		if (dasasreso > Vlimits[0] - 200) {
			// Yellow color
			color_fs = vec4(1.0, 1.0, 0.0, 1.0);
			// Size
			float PSCALE = 20.0;

			ssd_coord = gs_in[0].asasreso + PSCALE * vec2(-1.000,  0.000);
			gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
			EmitVertex();
			ssd_coord = gs_in[0].asasreso + PSCALE * vec2( 0.000, -1.000);
			gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
			EmitVertex();
			ssd_coord = gs_in[0].asasreso + PSCALE * vec2( 0.000,  1.000);
			gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
			EmitVertex();
			ssd_coord = gs_in[0].asasreso + PSCALE * vec2( 1.000,  0.000);
			gl_Position = gl_in[0].gl_Position + VSCALE * vec4(gs_in[0].vAR * ssd_coord, 0.0, 0.0);
			EmitVertex();
			EndPrimitive();
		}
    }
}
