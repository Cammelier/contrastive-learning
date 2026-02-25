import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.common.config import load_config
from src.common.logging import setup_logger
from src.data.io import load_df, save_df
from src.data.preprocessing import (
    QuantileClipper,
    TopNCategoryEncoder,
    drop_nans,
    ml_split,
    query_filter,
    rare_category_filter,
)

setup_logger()
logger = logging.getLogger(__name__)

def preprocess_df(
    df: pd.DataFrame,
    cfg: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe for machine learning tasks.
    Mantiene tutte le classi presenti nel dataset originale.
    """
    # Estrazione parametri dal config
    num_cols = list(cfg.data.num_cols)
    cat_cols = list(cfg.data.cat_cols)
    label_col = cfg.data.label_col
    benign_tag = cfg.data.benign_tag
    query = cfg.data.filter_query
    
    # Parametri critici per non perdere classi
    # Se min_cat_count è > 1, le classi rare spariscono!
    min_cat_count = cfg.data.get('min_cat_count', 1) 
    
    top_n_categories = cfg.data.top_n_categories
    train_frac = cfg.data.train_frac
    val_frac = cfg.data.val_frac
    test_frac = cfg.data.test_frac
    random_state = cfg.seed

    # 1. Pulizia iniziale
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    
    # 2. Filtro query (se definito nel yaml)
    if query:
        df = query_filter(df, query)
    
    # 3. Filtro categorie rare (impostato a 1 per non eliminare classi di attacco)
    logger.info(f"Applicazione rare_category_filter con min_count={min_cat_count}")
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    # Verifica classi presenti prima dello split
    unique_labels = df[label_col].unique()
    logger.info(f"Classi totali rilevate prima dello split ({len(unique_labels)}): {unique_labels}")

    # 4. Split del dataset
    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )

    # Configurazione Sklearn per restituire DataFrame
    set_config(transform_output="pandas")

    # Pipeline Numerica
    num_pipeline = Pipeline(
        steps=[
            ("quantile_clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline Categorica
    cat_pipeline = Pipeline(
        steps=[
            ("top_n_encoder", TopNCategoryEncoder(top_n=top_n_categories)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Fit e Transform
    logger.info("Fitting transformers...")
    preprocessor.fit(train_df)
    train_df = preprocessor.transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # 5. Encoding delle Label (Cruciale per le 10 classi)
    label_encoder = LabelEncoder()
    multi_label_col = f"multi_{label_col}"
    
    # Fit sul dataset intero per essere sicuri di mappare tutte le classi
    label_encoder.fit(df[label_col])
    
    train_df[multi_label_col] = label_encoder.transform(train_df[label_col])
    val_df[multi_label_col] = label_encoder.transform(val_df[label_col])
    test_df[multi_label_col] = label_encoder.transform(test_df[label_col])

    # 6. Label Binaria (opzionale, utile per task secondari)
    if benign_tag is not None:
        bin_label_col = f"bin_{label_col}"
        train_df[bin_label_col] = (train_df[label_col] != benign_tag).astype(int)
        val_df[bin_label_col] = (val_df[label_col] != benign_tag).astype(int)
        test_df[bin_label_col] = (test_df[label_col] != benign_tag).astype(int)

    label_mapping = {
        int(i): str(class_name) for i, class_name in enumerate(label_encoder.classes_)
    }
    logger.info(f"Label mapping finale: {label_mapping}")

    # 7. Salvataggio Metadati
    metadata = {
        "label_mapping": label_mapping,
        "num_classes": len(label_encoder.classes_),
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(df),
        },
        "samples_per_class": {
            "train": train_df[label_col].value_counts().to_dict(),
            "test": test_df[label_col].value_counts().to_dict(),
        },
    }

    metadata_path = Path(cfg.path.processed_data) / "df_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return train_df, val_df, test_df

def main() -> None:
    """Main entry point per la preparazione dati."""
    cfg = load_config(
        config_path=Path(__file__).parent / "config",
        config_name="config",
        overrides=sys.argv[1:],
    )

    logger.info(f"Lavoro su dataset: {cfg.data.file_name}")
    raw_data_path = Path(cfg.path.raw_data) / f"{cfg.data.file_name}.csv"
    
    if not raw_data_path.exists():
        logger.error(f"File non trovato: {raw_data_path}")
        return

    # Caricamento
    df = load_df(str(raw_data_path))

    # Preprocessing
    train_df, val_df, test_df = preprocess_df(df, cfg)

    # Salvataggio Split
    processed_data_dir = Path(cfg.path.processed_data)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_path = processed_data_dir / f"{cfg.data.file_name}_{split_name}.{cfg.data.extension}"
        save_df(split_df, str(output_path))
        logger.info(f"Salvato {split_name}: {len(split_df)} campioni")

    logger.info("Preparazione dati completata con successo!")

if __name__ == "__main__":
    main()