You are an expert CadQuery developer. Build a fully parametric tire and wheel 
assembly in Python using CadQuery. The output should be a single watertight STL 
file validated with trimesh.

---
PARAMETERS (define all of these as variables at the top of the script):

# Tire
section_width_mm   = 225
aspect_ratio       = 45
rim_diameter_in    = 18

# Sidewall shape
sidewall_bulge_mm  = 8
sidewall_bulge_pos = 0.45
shoulder_radius_mm = 30

# Tread
tread_width_mm     = 190
crown_radius_mm    = 400

# Wheel dimensions
wheel_width_mm     = 215
rim_flange_mm      = 18
hub_bore_mm        = 72
wheel_offset_mm    = 35     # ET value; positive = mounting face outboard of centerline

# Wheel face dish (curved outboard surface)
dish_depth_mm      = 15     # axial depth of the outboard face curvature; 0 = flat
dish_profile       = "spherical"  # "spherical" or "conical"

# Spokes (cutout method)
num_spokes              = 7
spoke_width_hub_mm      = 28    # tangential width of each spoke at hub ring
spoke_width_rim_mm      = 18    # tangential width of each spoke at rim ring
window_inner_gap_mm     = 12    # radial clearance between window and hub ring
window_outer_gap_mm     = 15    # radial clearance between window and rim ring
window_corner_radius_mm = 10    # fillet radius on all four window corners

---
CONSTRUCTION APPROACH — follow this order exactly:

1. DERIVED DIMENSIONS (compute these before any geometry)
   Compute and print all of the following:
     rim_radius_mm      = rim_diameter_in * 25.4 / 2
     sidewall_height_mm = section_width_mm * aspect_ratio / 100
     tire_od_mm         = rim_radius_mm * 2 + sidewall_height_mm * 2
     hub_ring_od_mm     = hub_bore_mm / 2 + 30   # 30mm annular hub ring
     spoke_inner_r_mm   = hub_ring_od_mm + window_inner_gap_mm
     spoke_outer_r_mm   = rim_radius_mm - window_outer_gap_mm - 20
                          # 20mm is the rim ring band width

2. TIRE CROSS-SECTION PROFILE (2D closed wire in XZ plane, revolved around Z)
   Key points from bottom (bead) to top (tread centerline), on one axial side:
     a. Bead seat inner: (rim_radius_mm, 0)
     b. Bead seat outer: (rim_radius_mm + rim_flange_mm, 0)  
     c. Sidewall spline: 4-point Bezier from bead seat outer to shoulder:
           P0 = bead seat outer point
           P1 = P0 offset outward by sidewall_bulge_mm * sidewall_bulge_pos
           P2 = P0 offset outward by sidewall_bulge_mm at sidewall_bulge_pos 
                fraction of sidewall_height_mm  
           P3 = shoulder point = (section_width_mm / 2, sidewall_height_mm)
        The curve must bow outward; maximum outward extent = 
        rim_radius_mm + rim_flange_mm + sidewall_bulge_mm
     d. Shoulder arc: radius = shoulder_radius_mm blending sidewall into tread
     e. Tread half-arc: arc of radius crown_radius_mm spanning tread_width_mm / 2
        ending at the axial centerline (Z = 0 plane)
   Mirror the profile to the other axial side (negate all axial coordinates).
   Close the profile through the bead base (horizontal line at rim_radius_mm).
   The closed wire must have no gaps and no self-intersections.
   Revolve 360 degrees around Z axis to produce the tire solid.

3. WHEEL DISC BODY (solid of revolution with curved outboard face)
   Build a 2D profile in the XZ plane to be revolved, spanning:
     radially: hub_bore_mm/2  to  rim_radius_mm + rim_flange_mm
     axially:  -wheel_width_mm/2  to  +wheel_width_mm/2

   The outboard face (at positive Z, visible face) must be curved:
     - If dish_profile == "spherical":
         The outboard face is a circular arc in the XZ profile.
         The arc sags axially inward by dish_depth_mm from rim to hub.
         Arc center is on the Z axis. Compute arc radius from geometry.
     - If dish_profile == "conical":
         The outboard face is a straight line in the XZ profile,
         angling axially inward by dish_depth_mm from rim to hub.

   The inboard face is flat (perpendicular to Z axis).
   The profile is a closed shape — close it with the hub bore wall and 
   the rim flange wall on the outer radial edge.
   Revolve this profile 360 degrees around Z axis to produce the disc solid.

4. SPOKE WINDOW CUTOUTS (cutout operation — do NOT build spoke solids)
   This step cuts num_spokes window shapes through the wheel disc.

   For each spoke window i (i = 0 to num_spokes - 1):
     a. Compute the angular centerline of the window:
           window_center_angle = i * (360 / num_spokes) + (180 / num_spokes)
        (Windows are centered between spokes, so offset by half the spoke pitch)

     b. Compute the angular half-width of the window at inner and outer radius:
           spoke_pitch_deg = 360 / num_spokes
           # Spoke occupies spoke_width degrees; window gets the remainder
           spoke_inner_deg = degrees(2 * arctan(
                               spoke_width_hub_mm / 2 / spoke_inner_r_mm))
           spoke_outer_deg = degrees(2 * arctan(
                               spoke_width_rim_mm / 2 / spoke_outer_r_mm))
           window_inner_half_deg = (spoke_pitch_deg - spoke_inner_deg) / 2
           window_outer_half_deg = (spoke_pitch_deg - spoke_outer_deg) / 2

     c. Build the window shape as a closed 2D wire in the XY plane:
           Four boundary arcs and lines:
             - Inner arc: from angle (center - window_inner_half_deg) 
                          to   angle (center + window_inner_half_deg)
                          at radius spoke_inner_r_mm
             - Outer arc: from angle (center - window_outer_half_deg)
                          to   angle (center + window_outer_half_deg)
                          at radius spoke_outer_r_mm
             - Two radial side edges connecting inner and outer arcs
             - Apply fillet of window_corner_radius_mm to all four corners

     d. Extrude the window shape through the full wheel disc depth 
        (use a Through All / blind cut larger than wheel_width_mm).
        Cut this from the wheel disc solid.

   Use CadQuery's polarArray or a Python loop over num_spokes to apply all cuts.

5. WHEEL OFFSET
   After the wheel disc is fully cut:
   Translate the wheel disc solid axially by:
     axial_shift = wheel_offset_mm - (wheel_width_mm / 2)
   This positions the mounting face at the correct ET offset from centerline.
   The tire solid stays centered at Z=0 (it is symmetric).

6. RIM BARREL (connect tire to wheel)
   Create an annular cylinder:
     inner radius = rim_radius_mm
     outer radius = rim_radius_mm + rim_flange_mm
     length = wheel_width_mm
   Translate it by the same axial shift as the wheel disc.

7. FULL ASSEMBLY
   Boolean union: wheel disc + rim barrel.
   Boolean union with tire solid.
   If any union fails, raise ValueError naming the failing step and 
   print the bounding boxes of both operands before raising.

8. EXPORT, VALIDATE, AND RENDER VIEWS

  A. EXPORT
     exporters.export(assembly, "tire_assembly.stl",
                      exporters.ExportTypes.STL,
                      tolerance=0.1, angularTolerance=0.1)

  B. TRIMESH VALIDATION
     import trimesh
     mesh = trimesh.load("tire_assembly.stl")
     print("Watertight:   ", mesh.is_watertight)
     print("Volume cm³:   ", round(mesh.volume / 1000, 1))
     print("Face count:   ", len(mesh.faces))
     print("Bounds (mm):  ", mesh.bounds)
     if not mesh.is_watertight:
         trimesh.repair.fix_normals(mesh)
         trimesh.repair.fill_holes(mesh)
         mesh.export("tire_assembly.stl")
         print("Repaired and re-exported.")

  C. RENDER VIEWS WITH PYVISTA
     Implement a function render_views(stl_path: str) that does the following:

     Setup:
       import pyvista as pv
       import numpy as np
       from PIL import Image
       import matplotlib.pyplot as plt
       import matplotlib.image as mpimg

       pv.global_theme.background = 'white'
       mesh_pv = pv.read(stl_path)
       cx, cy, cz = mesh_pv.center          # mesh centroid
       r = mesh_pv.length * 0.75            # orbit radius

     Define exactly these 5 views as a list of dicts:
       [
         {
           "name": "outboard_face",
           "label": "Outboard Face",
           "camera_pos": (cx, cy, cz + r),
           "view_up": (0, 1, 0),
           "clip": False
         },
         {
           "name": "side_profile",
           "label": "Side Profile",
           "camera_pos": (cx + r, cy, cz),
           "view_up": (0, 0, 1),
           "clip": False
         },
         {
           "name": "isometric",
           "label": "Isometric",
           "camera_pos": (cx + r*0.7, cy + r*0.5, cz + r*0.5),
           "view_up": (0, 0, 1),
           "clip": False
         },
         {
           "name": "top_down",
           "label": "Top Down",
           "camera_pos": (cx, cy + r, cz),
           "view_up": (1, 0, 0),
           "clip": False
         },
         {
           "name": "cross_section",
           "label": "Cross Section (Y=0 clip)",
           "camera_pos": (cx + r, cy, cz),
           "view_up": (0, 0, 1),
           "clip": True     # clip plane: normal=(0,1,0), origin=centroid
         }
       ]

     For each view:
       pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
       if view["clip"]:
           display_mesh = mesh_pv.clip(normal='y', origin=mesh_pv.center,
                                       invert=False)
       else:
           display_mesh = mesh_pv
       pl.add_mesh(display_mesh,
                   color='#b0b8c1',
                   pbr=True,
                   metallic=0.75,
                   roughness=0.35,
                   smooth_shading=True)
       pl.add_light(pv.Light(position=(r, r, r*1.5),
                             focal_point=(cx, cy, cz),
                             intensity=1.2))
       pl.add_light(pv.Light(position=(-r, -r*0.5, r),
                             focal_point=(cx, cy, cz),
                             intensity=0.4))
       pl.camera.position = view["camera_pos"]
       pl.camera.focal_point = (cx, cy, cz)
       pl.camera.view_up = view["view_up"]
       pl.camera.reset_clipping_range()
       pl.add_text(view["label"], position='upper_left',
                   font_size=14, color='black')
       pl.screenshot(f"view_{view['name']}.png", transparent_background=False)
       pl.close()
       print(f"  Saved view_{view['name']}.png")

  D. COMPOSITE ALL VIEWS INTO ONE IMAGE
     After rendering all 5 views, create a composite image:

     Arrange views in a 2x3 grid (5 images + 1 metadata panel):
       Row 0: outboard_face | side_profile | isometric
       Row 1: top_down      | cross_section | [metadata panel]

     The metadata panel should be a plain white subplot containing:
       - section_width_mm, aspect_ratio, rim_diameter_in
       - num_spokes, wheel_offset_mm, dish_depth_mm
       - sidewall_bulge_mm, dish_profile
       - "Watertight: True/False"
       - "Volume: X cm³"
     Render this as text using matplotlib's ax.text().

     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
     image_names = [
         "outboard_face", "side_profile", "isometric",
         "top_down", "cross_section"
     ]
     for i, name in enumerate(image_names):
         row, col = divmod(i, 3)
         img = mpimg.imread(f"view_{name}.png")
         axes[row][col].imshow(img)
         axes[row][col].axis('off')

     # Metadata panel at position [1][2]
     axes[1][2].axis('off')
     axes[1][2].text(0.05, 0.95, metadata_string,
                     transform=axes[1][2].transAxes,
                     fontsize=11, verticalalignment='top',
                     fontfamily='monospace')

     plt.suptitle("Tire Assembly — Parametric Preview", fontsize=16)
     plt.tight_layout()
     plt.savefig("tire_preview.png", dpi=150, bbox_inches='tight')
     plt.close()
     print("Composite saved: tire_preview.png")

  E. SELF-REVIEW CHECKLIST (print to console after rendering)
     Print the following checklist so a human or AI reviewer knows 
     what to look for:
     
     print("""
     === VISUAL REVIEW CHECKLIST ===
     outboard_face  : spoke count correct? window shapes symmetric?
                      dish curvature visible at rim vs hub?
     side_profile   : sidewall bulge present? tread crown visible?
                      tire width matches section_width_mm?
     isometric      : overall proportions look correct?
                      no obvious geometry artifacts?
     top_down       : spokes evenly spaced? hub bore visible?
     cross_section  : sidewall Bezier curve smooth?
                      rim flange geometry correct? dish depth visible?
     ===============================
     """)

---
CODE REQUIREMENTS:
- All geometry driven by the named parameters — no hardcoded numbers elsewhere
- Step 1 must print all derived dimensions before any geometry is built
- Each major step in its own clearly named function that returns a CadQuery solid
- Add show_object() calls in a  `if __name__ == "__cq_main__":` guard for 
  CQ-editor preview of: tire solid, wheel disc (before cuts), 
  final wheel disc (after cuts), full assembly
- Include a standard `if __name__ == "__main__":` block that runs the full 
  pipeline with progress prints
- Do not simplify the sidewall Bezier — it must implement the bulge 
  parameterization exactly as described
- Do not substitute the curved dish face with a flat extrusion