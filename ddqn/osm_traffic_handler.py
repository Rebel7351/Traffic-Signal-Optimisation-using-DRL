import os
import glob
import xml.etree.ElementTree as ET
import sys
import gzip
import shutil



class Traffic:
    def __init__(self,osm_dir):
        # self.osm_dir='D:\\mtech\\projects\\Thesis Work\\4thSem\\TrafficSignalOptimization\\experiments\\real_traffic\\2025-09-13-17-50-51'

        self.osm_dir=osm_dir
        self.net_file=str
        self.route_file=str
        
    def _extract_gz_files(self):
        """Extract all .gz files in the OSM directory"""
        print(f"Extracting compressed files in {self.osm_dir}...")
        
        gz_files = glob.glob(os.path.join(self.osm_dir, "osm.net.xml.gz"))
        
        for gz_file in gz_files:
            # Get the output filename (remove .gz extension)
            output_file = gz_file[:-3]
            
            print(f"  Extracting {os.path.basename(gz_file)} -> {os.path.basename(output_file)}")
            
            try:
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"  Error extracting {gz_file}: {e}")
                return False
        
        print(f"Extracted {len(gz_files)} compressed files")
        return True
    

    def _parse_sumo_config(self,config_file):
        """Parse SUMO configuration file and extract file paths"""
        print(f"📄 Parsing SUMO configuration: {os.path.basename(config_file)}")
        
        try:
            tree = ET.parse(config_file)
            root = tree.getroot()
            
            config_dir = os.path.dirname(os.path.abspath(config_file))
            
            # Find input section
            input_section = root.find('input')
            if input_section is None:
                # Try to find elements directly under root
                input_section = root
            
            parsed_config = {
                'net_file': None,
                'route_files': [],
                'additional_files': [],
                'config_dir': config_dir
            }
            
            # Extract network file
            net_elem = input_section.find('net-file')
            if net_elem is not None:
                net_file = net_elem.get('value') or net_elem.get('v')
                if net_file:
                    if not os.path.isabs(net_file):
                        net_file = os.path.join(config_dir, net_file)
                    parsed_config['net_file'] = net_file
            
            # Extract route files
            route_elem = input_section.find('route-files')
            if route_elem is not None:
                route_files_str = route_elem.get('value') or route_elem.get('v')
                if route_files_str:
                    route_files = [f.strip() for f in route_files_str.split(',')]
                    for route_file in route_files:
                        if route_file:
                            if not os.path.isabs(route_file):
                                route_file = os.path.join(config_dir, route_file)
                            parsed_config['route_files'].append(route_file)
            
            # Extract additional files (optional)
            add_elem = input_section.find('additional-files')
            if add_elem is not None:
                add_files_str = add_elem.get('value') or add_elem.get('v')
                if add_files_str:
                    add_files = [f.strip() for f in add_files_str.split(',')]
                    for add_file in add_files:
                        if add_file:
                            if not os.path.isabs(add_file):
                                add_file = os.path.join(config_dir, add_file)
                            parsed_config['additional_files'].append(add_file)
            
            return parsed_config
            
        except Exception as e:
            print(f"❌ Error parsing SUMO config: {e}")
            return None
        

    def _create_combined_route_file(self,route_files, output_file):
        """Create a combined route file from multiple route files"""
        print(f"🔄 Creating combined route file: {os.path.basename(output_file)}")
        
        # First, let's run duarouter to convert any .trips.xml files to .rou.xml
        converted_files = []
        net_file = None
        
        # Find network file from the same directory
        config_dir = os.path.dirname(route_files[0]) if route_files else "."
        net_files = glob.glob(os.path.join(config_dir, "*.net.xml*"))
        if net_files:
            net_file = net_files[0]
        
        for route_file in route_files:
            if route_file.endswith('.trips.xml') and net_file:
                # Convert trips to routes
                converted_file = route_file.replace('.trips.xml', '.rou.xml')
                if not os.path.exists(converted_file):
                    print(f"  Converting {os.path.basename(route_file)} to routes...")
                    import subprocess
                    cmd = [
                        "duarouter",
                        "--net-file", net_file,
                        "--trip-files", route_file,
                        "--output-file", converted_file,
                        "--ignore-errors", "true",
                        "--no-warnings", "true"
                    ]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        converted_files.append(converted_file)
                        print(f"    ✅ Converted to {os.path.basename(converted_file)}")
                    except Exception as e:
                        print(f"    ❌ Failed to convert: {e}")
                        # Use original file anyway
                        converted_files.append(route_file)
                else:
                    converted_files.append(converted_file)
            else:
                converted_files.append(route_file)
        
        # Now combine the route files
        root = ET.Element("routes")
        
        # Add default vehicle type
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "mixed_traffic")
        vtype.set("vClass", "passenger")
        vtype.set("color", "yellow")
        
        all_elements = []
        
        for route_file in converted_files:
            if not os.path.exists(route_file):
                print(f"  ⚠️  Warning: {route_file} not found, skipping...")
                continue
            
            print(f"  Processing {os.path.basename(route_file)}")
            
            try:
                tree = ET.parse(route_file)
                file_root = tree.getroot()
                
                for elem in file_root:
                    if elem.tag in ['vehicle', 'route', 'flow', 'vType']:
                        # Ensure vehicles have a type
                        if elem.tag in ['vehicle', 'flow'] and 'type' not in elem.attrib:
                            elem.set('type', 'mixed_traffic')
                        all_elements.append(elem)
            
            except Exception as e:
                print(f"    ❌ Error processing {route_file}: {e}")
        
        # Add all elements to root
        for elem in all_elements:
            root.append(elem)
        
        # Write combined file
        try:
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)
            tree.write(output_file, encoding='UTF-8', xml_declaration=True)
            
            print(f"  ✅ Combined route file created with {len(all_elements)} elements")
            return True
            
        except Exception as e:
            print(f"  ❌ Error writing combined file: {e}")
            return False


    def _find_sumo_config_files(self):
        """Find SUMO configuration and related files"""
        files = {
            'config_file': None,
            'net_file': None
        }
        
        # Find SUMO configuration file
        config_files = glob.glob(os.path.join(self.osm_dir, "*.sumocfg"))
        print(f'----------{config_files}')
        if config_files:
            files['config_file'] = config_files[0]
        
        # Find network file (compressed or uncompressed)
        net_files = glob.glob(os.path.join(self.osm_dir, "*.net.xml"))
        if not net_files:
            net_files = glob.glob(os.path.join(self.osm_dir, "*.net.xml.gz"))
        if net_files:
            files['net_file'] = net_files[0]
        
        return files

    def preprocessing(self):
        if not os.path.isfile(os.path.join(self.osm_dir,'combined_from_config.rou.xml')) \
             or not os.path.isfile(os.path.join(self.osm_dir,'osm.net.xml')):
            
            self._extract_gz_files()    
            osm_files = self._find_sumo_config_files()
            sumo_config = osm_files['config_file']
            
            if not sumo_config:
                print(f"❌ No SUMO configuration file (.sumocfg) found in {self.osm_dir}")
                sys.exit(1)
            
            print(f"Found SUMO config: {os.path.basename(sumo_config)}")
        

            if not os.path.exists(sumo_config):
                print(f"❌ SUMO config file not found: {sumo_config}")
                sys.exit(1)
            
            # Parse the SUMO configuration
            config_data = self._parse_sumo_config(sumo_config)
            if not config_data:
                print("❌ Failed to parse SUMO configuration file!")
                sys.exit(1)
            
            self.net_file = config_data['net_file']
            route_files = config_data['route_files']
            
            print(f"📁 Network file: {os.path.basename(self.net_file) if self.net_file else 'None'}")
            print(f"📁 Route files: {len(route_files)} files")
            for rf in route_files[:5]:  # Show first 5
                print(f"  - {os.path.basename(rf)}")
            if len(route_files) > 5:
                print(f"  ... and {len(route_files) - 5} more")
            
            if not self.net_file or not route_files:
                print("❌ Required files not found in SUMO configuration!")
                sys.exit(1)
            
            # Create combined route file
            self.route_file = os.path.join(self.osm_dir, "combined_from_config.rou.xml")
            if not self._create_combined_route_file(route_files, self.route_file ):
                print("❌ Failed to create combined route file!")
                sys.exit(1)
            

   