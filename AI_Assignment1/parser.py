import numpy as np

def parse_tsplib(file_path):
    """This parses the TSPLIB files and extracts city coordinates."""
    data = {}
    node_coords = []
    reading_nodes = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line == "EOF":
                break

            if reading_nodes:
                parts = line.split()
                if len(parts) == 3:
                    node_coords.append((int(parts[0]), float(parts[1]), float(parts[2])))
                continue

            key_value = line.split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                data[key.strip()] = value.strip()
            elif line == "NODE_COORD_SECTION":
                reading_nodes = True

    
    data["NODE_COORDS"] = np.array(node_coords)[:, 1:] 
    return data
