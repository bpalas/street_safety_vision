import json
import pandas as pd

# Paths
jsonl_file_path = "data/json/batch_vwx6bfxuDSrlSctyJrMgionx_output.jsonl"
csv_input_path = "data/raw/delitos_imagenes_santiago_traducido.csv"
csv_output_path = "data/inferences/delitos_imagenes_descripcion_santiago.csv"

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def extract_results(json_lines):
    results_list = []
    for res in json_lines:
        task_id = res.get('custom_id', '')
        index = int(task_id.split('-')[-1]) if '-' in task_id else None
        result = res.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
        results_list.append({
            'index': index,
            'nombre_foto': res.get('input', {}).get('nombre_foto', ''),
            'result': result
        })
    return pd.DataFrame(results_list)

def main():
    # Leer JSONL y extraer resultados
    json_lines = read_jsonl(jsonl_file_path)
    results_df = extract_results(json_lines)

    # Leer CSV original
    df = pd.read_csv(csv_input_path)

    # Unir por 'nombre_foto'
    df_merged = df.merge(results_df[['nombre_foto', 'result']], on='nombre_foto', how='left')

    # Guardar CSV final
    df_merged.to_csv(csv_output_path, index=False)
    print(f"Archivo CSV guardado en: {csv_output_path}")

if __name__ == "__main__":
    main()