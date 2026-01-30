import nd2
import os
import json

'''
Pull metadata for initial input GUIs.
'''
def nd2_metadata_processor(filepath):
    with nd2.ND2File(filepath) as f:
        metadata = f.metadata
        sizes = f.sizes
        shape = f.shape
        
        # Access channels as attribute (likely a list)
        channels = getattr(metadata, 'channels', [])
        # If channel is a list of channel objects, get their names
        c_list = [ch.channel.name for ch in metadata.channels]
        
        c_num = sizes.get('C', 1)
        z_num = sizes.get('Z', sizes.get('T', 1))
        y_num = sizes.get('Y', 1)
        x_num = sizes.get('X', 1)
        
        return {
            'c_num': c_num,
            'z_num': z_num,
            'c_list': c_list,
            'shape': shape,
            'sizes': sizes,
            'height': y_num,
            'width': x_num
        }


'''
Following functions are for generating Metadata JSON.
'''
def extract_nd2_metadata_nd2lib(nd2_path, json_out_dir):
    if not os.path.isdir(json_out_dir):
        raise ValueError(f"Expected output directory, got {json_out_dir}")

    # Construct JSON filename based on ND2 filename
    base_name = os.path.splitext(os.path.basename(nd2_path))[0]
    json_out_path = os.path.join(json_out_dir, f"{base_name}_nd2metadata.json")

    with nd2.ND2File(nd2_path) as myfile:
        attributes_dict = myfile.attributes._asdict() if hasattr(myfile.attributes, '_asdict') else vars(myfile.attributes)
        frame_count = myfile._frame_count

        all_frame_metadata = [
            make_json_serializable(myfile.frame_metadata(i)) for i in range(frame_count)
        ]

        combined_data = {
            "attributes": make_json_serializable(attributes_dict),
            "metadata": make_json_serializable(myfile.metadata),
            "experiment": make_json_serializable(myfile.experiment),
            "text_info": make_json_serializable(myfile.text_info),
            "events": make_json_serializable(myfile.events()),
            "frame_metadata": all_frame_metadata
        }

        with open(json_out_path, "w") as f:
            json.dump(combined_data, f, indent=2)

    print(f"Metadata and attributes written to {json_out_path}")

# Helper from above
def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if hasattr(obj, "_asdict"):
        return make_json_serializable(obj._asdict())
    if hasattr(obj, "__dict__"):
        return make_json_serializable(vars(obj))
    return str(obj)
