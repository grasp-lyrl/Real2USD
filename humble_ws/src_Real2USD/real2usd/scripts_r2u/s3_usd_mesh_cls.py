import boto3
import tempfile
import os
from pxr import Usd
from urllib.parse import urlparse
from pxr import Usd, UsdGeom, Gf
from matplotlib import pyplot as plt
from ipdb import set_trace as st
import seaborn as sns
sns.set_theme()
fsz = 24
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.5*fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})
import numpy as np

class s3_usd_interaction:
    def __init__(self):
        os.environ["AWS_ACCESS_KEY_ID"] = ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = ""
        os.environ["AWS_REGION"] = ""

        # Initialize Boto3 client
        self.s3 = boto3.client('s3')

    def load_local_usd(self, file_path):
        """
        Load a USD file from local directory
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        try:
            print(f"Loading local USD file: {file_path}")
            stage = Usd.Stage.Open(file_path)
            if stage:
                print("Successfully opened local USD stage")
                return stage
            else:
                print("Failed to open local USD stage")
                return None
        except Exception as e:
            print(f"Error opening local USD file: {e}")
            return None

    def download_usd(self, s3_uri):
        """
        download usd file as a temp file from the s3 uri and load it as a USD stage
        """

        # Parse the S3 URI (uniform resource identifier)
        parsed_s3 = urlparse(s3_uri)
        if parsed_s3.scheme != "s3":
            raise ValueError("Provided URI is not a valid S3 URI.")

        bucket_name = parsed_s3.netloc # The bucket name
        object_key = parsed_s3.path.lstrip("/") # The object key

        # print(f"Bucket Name: {bucket_name}")
        # print(f"Object Key: {object_key}")

        # Download the file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as temp_file:
            try:
                # Fetch the object from S3
                self.s3.download_fileobj(bucket_name, object_key, temp_file)
                
                # Ensure data is written to disk
                temp_file.flush()
                temp_file.close()
                
                # Try to load the USD file using Usd.Stage.Open
                stage = Usd.Stage.Open(temp_file.name)
                if stage:
                    print(f"Successfully opened USD stage: {temp_file.name}")
                    return stage
                else:
                    print("Failed to open USD stage.")
                    return None
                    
            except Exception as e:
                print(f"Error downloading or opening USD file: {e}")
                return None
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def download_usd_with_payloads(self, s3_uri):
        """
        Alternative method for USD files with payload dependencies
        Downloads to a persistent location and tries to resolve dependencies
        """
        parsed_s3 = urlparse(s3_uri)
        bucket_name = parsed_s3.netloc
        object_key = parsed_s3.path.lstrip("/")
        
        # Create a more permanent download location
        download_dir = os.path.join(os.getcwd(), "temp_usd_downloads")
        os.makedirs(download_dir, exist_ok=True)
        
        filename = os.path.basename(object_key)
        local_path = os.path.join(download_dir, filename)
        
        try:
            print(f"Downloading to persistent location: {local_path}")
            self.s3.download_file(bucket_name, object_key, local_path)
            
            # Try to open with different options
            stage_options = [
                Usd.Stage.Open(local_path),
                Usd.Stage.Open(local_path, Usd.Stage.LoadNone),
                Usd.Stage.Open(local_path, Usd.Stage.LoadAll)
            ]
            
            for i, stage in enumerate(stage_options):
                if stage:
                    print(f"Successfully opened USD stage with method {i+1}")
                    return stage
            
            print("All loading methods failed")
            return None
            
        except Exception as e:
            print(f"Error in persistent download method: {e}")
            return None

    def inspect_usd_stage(self, stage, max_depth=5):
        """
        Debug function to inspect the USD stage structure
        """
        print(f"\n=== USD Stage Inspection ===")
        print(f"Stage root: {stage.GetRootLayer().identifier}")
        print(f"Stage default prim: {stage.GetDefaultPrim()}")
        
        def traverse_and_print(prim, depth=0, max_depth=max_depth):
            if depth > max_depth:
                return
                
            indent = "  " * depth
            prim_type = prim.GetTypeName()
            prim_path = prim.GetPath()
            
            # Check if it's a geometry type
            is_geom = prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.Curves)
            geom_marker = " [GEOMETRY]" if is_geom else ""
            
            # Check for payloads and references
            payload_info = ""
            ref_info = ""
            if prim.HasPayload():
                payload_info = " [HAS_PAYLOAD]"
            if prim.GetReferences():
                ref_info = " [HAS_REFERENCES]"
            
            print(f"{indent}{prim_type}: {prim_path}{geom_marker}{payload_info}{ref_info}")
            
            # Print additional info for geometry prims
            if is_geom:
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    if points_attr.HasAuthoredValue():
                        points = points_attr.Get()
                        print(f"{indent}  Points: {len(points)}")
                    else:
                        print(f"{indent}  Points: No authored value")
                        
                    # Check for payloads
                    if prim.HasPayload():
                        print(f"{indent}  Has payload: True")
                        
            # Recurse for children
            for child in prim.GetChildren():
                traverse_and_print(child, depth + 1, max_depth)
        
        # Start traversal from root
        root_prim = stage.GetPseudoRoot()
        traverse_and_print(root_prim)

    def load_payloads_and_references(self, stage):
        """
        Try to load payloads and references to access hidden geometry
        """
        print(f"\n=== Loading Payloads and References ===")
        
        def load_prim_contents(prim, depth=0, max_depth=5):
            if depth > max_depth:  # Increase recursion limit
                return
                
            indent = "  " * depth
            prim_path = prim.GetPath()
            
            # Try to load payloads
            if prim.HasPayload():
                print(f"{indent}Loading payload for: {prim_path}")
                try:
                    # Force load the payload
                    prim.Load()
                    print(f"{indent}  Payload loaded successfully")
                except Exception as e:
                    print(f"{indent}  Failed to load payload: {e}")
            
            # Try to load references
            if prim.GetReferences():
                print(f"{indent}Loading references for: {prim_path}")
                try:
                    # Force load references
                    prim.Load()
                    print(f"{indent}  References loaded successfully")
                except Exception as e:
                    print(f"{indent}  Failed to load references: {e}")
            
            # Recurse for children
            for child in prim.GetChildren():
                load_prim_contents(child, depth + 1, max_depth)
        
        # Start from root and load all payloads/references
        root_prim = stage.GetPseudoRoot()
        load_prim_contents(root_prim)
        
        # Also try to load sublayers
        print(f"\n=== Checking Sublayers ===")
        root_layer = stage.GetRootLayer()
        sublayers = root_layer.subLayerPaths
        if sublayers:
            print(f"Found {len(sublayers)} sublayers:")
            for i, sublayer in enumerate(sublayers):
                print(f"  Sublayer {i+1}: {sublayer}")
                try:
                    # Try to load the sublayer
                    stage.GetLayerStack()
                    print(f"    Sublayer loaded successfully")
                except Exception as e:
                    print(f"    Failed to load sublayer: {e}")
        else:
            print("No sublayers found")
        
        print("Payload and reference loading completed")

    def inspect_references_in_detail(self, stage):
        """
        Inspect references in detail to see what files they point to
        """
        print(f"\n=== Detailed Reference Inspection ===")
        
        def inspect_prim_references(prim, depth=0, max_depth=5):
            if depth > max_depth:
                return
                
            indent = "  " * depth
            prim_path = prim.GetPath()
            
            # Check references
            if prim.GetReferences():
                print(f"{indent}Prim: {prim_path}")
                try:
                    # Get the reference info using the correct USD API
                    ref_info = prim.GetReferences()
                    
                    # Try to get the actual reference paths
                    if hasattr(ref_info, 'GetItems'):
                        items = ref_info.GetItems()
                        for i, item in enumerate(items):
                            print(f"{indent}    Reference {i+1}: {item}")
                            # Try to extract asset path
                            if hasattr(item, 'assetPath'):
                                print(f"{indent}      Asset path: {item.assetPath}")
                            if hasattr(item, 'primPath'):
                                print(f"{indent}      Prim path: {item.primPath}")
                    elif hasattr(ref_info, 'GetAll'):
                        items = ref_info.GetAll()
                        for i, item in enumerate(items):
                            print(f"{indent}    Reference {i+1}: {item}")
                            # Try to extract asset path
                            if hasattr(item, 'assetPath'):
                                print(f"{indent}      Asset path: {item.assetPath}")
                            if hasattr(item, 'primPath'):
                                print(f"{indent}      Prim path: {item.primPath}")
                    else:
                        # Try alternative approach - get the reference layer
                        try:
                            ref_layer = prim.GetReferences().GetLayer()
                            if ref_layer:
                                print(f"{indent}    Reference layer: {ref_layer.identifier}")
                        except:
                            print(f"{indent}    Reference details: {ref_info}")
                        
                except Exception as e:
                    print(f"{indent}    Could not extract reference details: {e}")
            
            # Check payloads
            if prim.HasPayload():
                print(f"{indent}Prim: {prim_path}")
                try:
                    payloads = prim.GetPayloads()
                    
                    # Try to get the actual payload paths
                    if hasattr(payloads, 'GetItems'):
                        items = payloads.GetItems()
                        for i, item in enumerate(items):
                            print(f"{indent}    Payload {i+1}: {item}")
                            # Try to extract asset path
                            if hasattr(item, 'assetPath'):
                                print(f"{indent}      Asset path: {item.assetPath}")
                            if hasattr(item, 'primPath'):
                                print(f"{indent}      Prim path: {item.primPath}")
                    elif hasattr(payloads, 'GetAll'):
                        items = payloads.GetAll()
                        for i, item in enumerate(items):
                            print(f"{indent}    Payload {i+1}: {item}")
                            # Try to extract asset path
                            if hasattr(item, 'assetPath'):
                                print(f"{indent}      Asset path: {item.assetPath}")
                            if hasattr(item, 'primPath'):
                                print(f"{indent}      Prim path: {item.primPath}")
                    else:
                        # Try alternative approach - get the payload layer
                        try:
                            payload_layer = prim.GetPayloads().GetLayer()
                            if payload_layer:
                                print(f"{indent}    Payload layer: {payload_layer.identifier}")
                        except:
                            print(f"{indent}    Payload details: {payloads}")
                        
                except Exception as e:
                    print(f"{indent}    Could not extract payload details: {e}")
            
            # Recurse for children
            for child in prim.GetChildren():
                inspect_prim_references(child, depth + 1, max_depth)
        
        # Start inspection from root
        root_prim = stage.GetPseudoRoot()
        inspect_prim_references(root_prim)

    def inspect_composition_arcs(self, stage):
        """
        Use USD composition query API to inspect references and payloads
        """
        print(f"\n=== Composition Arcs Inspection ===")
        
        def inspect_prim_composition(prim, depth=0, max_depth=5):
            if depth > max_depth:
                return
                
            indent = "  " * depth
            prim_path = prim.GetPath()
            
            # Check references using composition query
            try:
                # Get all composition arcs
                comp_query = prim.GetCompositionQuery()
                if comp_query:
                    print(f"{indent}Prim: {prim_path}")
                    
                    # Check for references
                    refs = comp_query.GetReferences()
                    if refs:
                        print(f"{indent}  References found: {len(refs)}")
                        for i, ref in enumerate(refs):
                            print(f"{indent}    Reference {i+1}: {ref}")
                    
                    # Check for payloads
                    payloads = comp_query.GetPayloads()
                    if payloads:
                        print(f"{indent}  Payloads found: {len(payloads)}")
                        for i, payload in enumerate(payloads):
                            print(f"{indent}    Payload {i+1}: {payload}")
                            
            except Exception as e:
                # Fallback to simpler method
                if prim.GetReferences():
                    print(f"{indent}Prim: {prim_path} [HAS_REFERENCES]")
                if prim.HasPayload():
                    print(f"{indent}Prim: {prim_path} [HAS_PAYLOAD]")
            
            # Recurse for children
            for child in prim.GetChildren():
                inspect_prim_composition(child, depth + 1, max_depth)
        
        # Start inspection from root
        root_prim = stage.GetPseudoRoot()
        inspect_prim_composition(root_prim)

    def load_usd_directly(self, usd_path):
        """
        Load USD file directly using the same approach as usd_buffer_node.py
        """
        print(f"\n=== Loading USD Directly (like usd_buffer_node.py) ===")
        
        try:
            stage = Usd.Stage.Open(usd_path)
            if not stage:
                print(f"Could not open USD stage at {usd_path}")
                return None

            print(f"✓ Successfully opened USD stage")
            
            # Get meters per unit
            meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
            print(f"Meters per unit: {meters_per_unit}")
            
            vertices = []
            
            # Get all mesh vertices using TraverseAll() like in usd_buffer_node.py
            for prim in stage.TraverseAll():
                if prim.IsA(UsdGeom.Mesh):
                    print(f"  Found mesh: {prim.GetPath()}")
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    if points_attr:
                        points = points_attr.Get()
                        if points:
                            print(f"    Points: {len(points)}")
                            
                            # Apply world transformation like in usd_buffer_node.py
                            xform = UsdGeom.Xformable(prim)
                            matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            
                            for p in points:
                                transformed_p = matrix.Transform(p)
                                scaled_p = transformed_p * meters_per_unit
                                vertices.append(scaled_p)
                        else:
                            print(f"    No points found")
                    else:
                        print(f"    No points attribute")
                elif prim.IsA(UsdGeom.Points):
                    print(f"  Found points: {prim.GetPath()}")
                    points_prim = UsdGeom.Points(prim)
                    points_attr = points_prim.GetPointsAttr()
                    if points_attr:
                        points = points_attr.Get()
                        if points:
                            print(f"    Points: {len(points)}")
                            xform = UsdGeom.Xformable(prim)
                            matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            
                            for p in points:
                                transformed_p = matrix.Transform(p)
                                scaled_p = transformed_p * meters_per_unit
                                vertices.append(scaled_p)
                elif prim.IsA(UsdGeom.Curves):
                    print(f"  Found curves: {prim.GetPath()}")
                    # Handle curves if needed

            vertices = np.array(vertices)
            print(f"Total vertices extracted: {len(vertices)}")
            
            if len(vertices) == 0:
                print("✗ No vertices found in USD file")
                return None
            else:
                print(f"✓ Successfully extracted {len(vertices)} vertices")
                
                # Calculate bounds
                min_bounds = np.min(vertices, axis=0)
                max_bounds = np.max(vertices, axis=0)
                center = (min_bounds + max_bounds) / 2.0
                extents = (max_bounds - min_bounds) / 2.0
                
                print(f"Bounds: min={min_bounds}, max={max_bounds}")
                print(f"Center: {center}")
                print(f"Extents: {extents}")
                
                return {
                    'vertices': vertices,
                    'center': center,
                    'extents': extents,
                    'stage': stage
                }
                
        except Exception as e:
            print(f"Error loading USD directly: {e}")
            return None

    def usd2meshpts(self, stage, render=False, max_points_per_mesh=None, skip_points=1, debug=False):
        """
        given a USD stage, extract the 3D mesh points in the world frame
        max_points_per_mesh: if set, limits the number of points visualized per mesh (None = no limit)
        skip_points: plot every Nth point (e.g., skip_points=100 plots every 100th point)
        """

        def find_geometry(stage, prim=None, indent=0, geometry_prims=None):
            """Recursively traverses the scene graph to find geometry nodes."""
            if prim is None:
                prim = stage.GetPseudoRoot()  # Start at the root if no prim is provided
                geometry_prims = []

            # Check if the current prim is a geometry node
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.Curves):
                if render:
                    print(" " * indent + f"Geometry found: {prim.GetPath()}")
                geometry_prims.append(prim)

            # Recurse for all children
            for child in prim.GetChildren():
                find_geometry(stage, child, indent + 2, geometry_prims)

            return geometry_prims
        

        mesh3d = {'vertices': {}, 'normals': {}}
        geometry_nodes = find_geometry(stage)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Output collected geometry paths
        total_points = 0
        for geom in geometry_nodes:
            mesh = UsdGeom.Mesh(geom)
            # Extract points (in local space)
            points = mesh.GetPointsAttr().Get()
            # Optionally apply world transformation
            transform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_points = [transform.Transform(point) for point in points]
            
            mesh3d['vertices'][geom.GetPath()] = world_points
            
            # get the normals of the vertcies if they exist
            if mesh.GetNormalsAttr().HasAuthoredValue():
                normals = mesh.GetNormalsAttr().Get()
                mesh3d['normals'][geom.GetPath()] = normals
            
            if render:
                # given these world points, plot them in 3D with matplotlib 
                itr = 0
                points_plotted = 0
                for i, point in enumerate(world_points):
                    # Only plot every Nth point based on skip_points parameter
                    if i % skip_points == 0:
                        ax.scatter(point[0], point[1], point[2], s=1, alpha=0.6)
                        points_plotted += 1
                    
                    itr += 1
                    total_points += 1
                    if max_points_per_mesh and itr > max_points_per_mesh:
                        print(f"  Limited visualization to {max_points_per_mesh} points for {geom.GetPath()}")
                        break   
                
                print(f"  Rendered {points_plotted} points from {geom.GetPath()} (total: {len(world_points)}, skip: every {skip_points}th point)")
        
        if render:
            print(f"\nTotal points processed: {total_points}")
            print(f"Points plotted with skip={skip_points}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.tight_layout()
            # ax.set_title(f'USD Mesh Visualization - {len(geometry_nodes)} geometry nodes (skip every {skip_points} points)')

        return mesh3d

    def run(self, uri_or_path, render=False, max_points_per_mesh=None, skip_points=1, debug=False):
        # Check if it's a local file path or S3 URI
        if uri_or_path.startswith('s3://'):
            # It's an S3 URI
            stage = self.download_usd(uri_or_path)
        else:
            # It's a local file path
            stage = self.load_local_usd(uri_or_path)
        
        if stage is None:
            print("Failed to load USD file")
            return None
        
        # Try the direct approach first (like usd_buffer_node.py)
        print(f"\n=== Trying Direct USD Loading ===")
        direct_result = self.load_usd_directly(uri_or_path)
        
        if direct_result and len(direct_result['vertices']) > 0:
            print(f"✓ Successfully extracted {len(direct_result['vertices'])} vertices directly!")
            
            # Create mesh3d structure from direct result
            mesh3d = {
                'vertices': {'direct_extraction': direct_result['vertices']},
                'normals': {}
            }
            
            if render:
                # Visualize the directly extracted vertices
                self.visualize_vertices(direct_result['vertices'], skip_points)
            
            return mesh3d
        else:
            print("Direct loading failed, falling back to original method...")
            
            # Debug: inspect the stage structure if requested
            if debug:
                self.inspect_usd_stage(stage)
                
            # Try to load payloads and references to access hidden geometry
            self.load_payloads_and_references(stage)
            
            # Debug: inspect again after loading payloads
            if debug:
                print(f"\n=== Stage Inspection After Loading Payloads ===")
                self.inspect_usd_stage(stage)
                
            mesh3d = self.usd2meshpts(stage, render=render, max_points_per_mesh=max_points_per_mesh, skip_points=skip_points, debug=debug)
            if render:
                plt.show()
            return mesh3d

    def visualize_vertices(self, vertices, skip_points=1):
        """
        Visualize the directly extracted vertices
        """
        print(f"\n=== Visualizing Directly Extracted Vertices ===")
        print(f"Total vertices: {len(vertices)}")
        print(f"Skip every {skip_points}th point")
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot every Nth point
        points_plotted = 0
        for i, point in enumerate(vertices):
            if i % skip_points == 0:
                ax.scatter(point[0], point[1], point[2], s=8, alpha=1)
                points_plotted += 1
        
        print(f"Plotted {points_plotted} points")
        
        # Set labels and title
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title(f'Directly Extracted USD Vertices ({points_plotted} points plotted)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    get_mesh = s3_usd_interaction()
    
    # Option 1: Load the actual geometry file directly (recommended)
    geometry_usd_path = "/data/SimReadyAssets/Curated_SimSearch/Chair/Chair/RoundedChair/3-LW-10-GI.usd"
    if os.path.exists(geometry_usd_path):
        print("=== Loading actual geometry file directly ===")
        mesh3d = get_mesh.run(geometry_usd_path, render=True, max_points_per_mesh=None, skip_points=50, debug=True)
    else:
        print(f"Geometry file not found: {geometry_usd_path}")
        
    
    if mesh3d is not None:
        print(f"Mesh data keys: {mesh3d.keys()}")
        print(f"Number of geometry nodes: {len(mesh3d['vertices'])}")
        
        # Print detailed info about each mesh
        for path, vertices in mesh3d['vertices'].items():
            print(f"  {path}: {len(vertices)} vertices")
    else:
        print("Failed to load any USD files")
        print("\nTo use local files:")
        print("1. Make sure the geometry file exists: Davis_Mez_Tables_MZ-1036-R_3D.usd")
        print("2. Or use the composition file: 0_roundtable-hallway.usd")
        print("3. Run the script again")

    

