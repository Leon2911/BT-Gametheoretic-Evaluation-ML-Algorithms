import glob
import os

from typing import Tuple

from PIL import Image

GRID_SHAPE: Tuple[int, int] = (10, 20)  # (Höhe, Breite) z.B. 10 Zeilen, 20 Spalten

# (Diese Werte müssen exakt mit deiner render_interactive_grid_replay Methode übereinstimmen)
HEATMAP_CELL_SIZE = 15
COLORBAR_HEIGHT = 10
TITLE_SPACE = 25
LABEL_SPACE = 20
SPACING_BETWEEN_HEATMAPS = 40
HEATMAP_X_START = 10  # Der X-Abstand vom Rand
HEATMAP_Y_START = 10  # Der Y-Abstand vom Rand
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ordner-Pfade
SOURCE_DIR = os.path.join(SCRIPT_DIR, "Ergebnisse", "Screenshots")
OUTPUT_KOOP_DIR = os.path.join(SOURCE_DIR, "Extracted_Heatmaps/Kooperationsrate")
OUTPUT_REWARD_DIR = os.path.join(SOURCE_DIR, "Extracted_Heatmaps/Rewardrate")


# --- 2. BERECHNUNG DER SCHNITTBEREICHE ---

def calculate_bounding_boxes(grid_shape):
    """
    Berechnet die exakten Pixel-Boxen für die beiden Heatmaps.
    """
    heatmap_height = grid_shape[0] * HEATMAP_CELL_SIZE
    heatmap_width = grid_shape[1] * HEATMAP_CELL_SIZE

    # Höhe eines kompletten Blocks (Titel + Legende + Labels + Gitter)
    single_heatmap_block_height = TITLE_SPACE + COLORBAR_HEIGHT + LABEL_SPACE + heatmap_height

    # --- KORREKTUR-OFFSET ---
    # Wenn das Bild zu tief ist, müssen wir y verringern.
    # Starte mit -25 Pixeln und passe an, falls nötig.
    OFFSET_Y_CORRECTION = -10

    # --- Box 1: Kooperationsrate (Oben) ---
    # Ursprüngliche Berechnung + Korrektur
    y1_start = HEATMAP_Y_START + TITLE_SPACE + LABEL_SPACE

    # Sicherheitscheck: y darf nicht negativ werden
    y1_start = max(0, y1_start)

    y1_end = y1_start + heatmap_height
    x1_start = HEATMAP_X_START
    x1_end = x1_start + heatmap_width

    box_koop = (x1_start, y1_start, x1_end, y1_end)

    # --- Box 2: Reward (Unten) ---
    reward_heatmap_y_top = HEATMAP_Y_START + single_heatmap_block_height + SPACING_BETWEEN_HEATMAPS

    # Auch hier die Korrektur anwenden
    y2_start = reward_heatmap_y_top + TITLE_SPACE + LABEL_SPACE + OFFSET_Y_CORRECTION
    y2_end = y2_start + heatmap_height
    x2_start = HEATMAP_X_START
    x2_end = x2_start + heatmap_width

    box_reward = (x2_start, y2_start, x2_end, y2_end)

    return box_koop, box_reward

#def calculate_bounding_boxes(grid_shape):
#    """
#    Berechnet die exakten Pixel-Boxen für die beiden Heatmaps.
#    """
#    heatmap_height = grid_shape[0] * HEATMAP_CELL_SIZE
#    heatmap_width = grid_shape[1] * HEATMAP_CELL_SIZE
#
#    # Höhe eines kompletten Blocks (Titel + Legende + Labels + Gitter)
#    single_heatmap_block_height = TITLE_SPACE + COLORBAR_HEIGHT + LABEL_SPACE + heatmap_height
#
#    # --- Box 1: Kooperationsrate (Oben) ---
#    # Die Heatmap selbst beginnt UNTER Titel und Label-Platz
#    y1_start = HEATMAP_Y_START + TITLE_SPACE + LABEL_SPACE
#    y1_end = y1_start + heatmap_height
#    x1_start = HEATMAP_X_START
#    x1_end = x1_start + heatmap_width
#
#    box_koop = (x1_start, y1_start, x1_end, y1_end)
#
#    # --- Box 2: Reward (Unten) ---
#    # Die zweite Box beginnt nach dem gesamten ersten Block
#    reward_heatmap_y_top = HEATMAP_Y_START + single_heatmap_block_height + SPACING_BETWEEN_HEATMAPS
#
#    y2_start = reward_heatmap_y_top + TITLE_SPACE + LABEL_SPACE
#    y2_end = y2_start + heatmap_height
#    x2_start = HEATMAP_X_START
#    x2_end = x2_start + heatmap_width
#
#    box_reward = (x2_start, y2_start, x2_end, y2_end)
#
#    return box_koop, box_reward


# --- 3. HAUPTFUNKTION ---

def extract_all_heatmaps():
    """
    Lädt alle Screenshots, schneidet die Heatmaps aus und speichert sie.
    """
    # Erstelle die Ausgabeordner
    os.makedirs(OUTPUT_KOOP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REWARD_DIR, exist_ok=True)

    # Berechne die Boxen
    box_koop, box_reward = calculate_bounding_boxes(GRID_SHAPE)

    print(f"Berechnete Koop-Box: {box_koop}")
    print(f"Berechnete Reward-Box: {box_reward}")

    # Finde alle Screenshot-Dateien
    search_path = os.path.join(SOURCE_DIR, "snapshot_match_*.png")
    screenshot_files = glob.glob(search_path)

    if not screenshot_files:
        print(f"Fehler: Keine Screenshot-Dateien in '{search_path}' gefunden.")
        return

    print(f"{len(screenshot_files)} Screenshots gefunden. Beginne Extraktion...")

    for filepath in screenshot_files:
        try:
            with Image.open(filepath) as img:
                # Extrahiere den Dateinamen für das neue Speichern
                filename = os.path.basename(filepath)

                # Schneide Box 1 (Kooperation) aus und speichere sie
                koop_heatmap = img.crop(box_koop)
                koop_heatmap.save(os.path.join(OUTPUT_KOOP_DIR, filename))

                # Schneide Box 2 (Reward) aus und speichere sie
                reward_heatmap = img.crop(box_reward)
                reward_heatmap.save(os.path.join(OUTPUT_REWARD_DIR, filename))

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {filepath}: {e}")

    print(f"Extraktion abgeschlossen. Ergebnisse in '{OUTPUT_KOOP_DIR}' und '{OUTPUT_REWARD_DIR}'.")


# --- 4. SKRIPT AUSFÜHREN ---
if __name__ == "__main__":
    extract_all_heatmaps()