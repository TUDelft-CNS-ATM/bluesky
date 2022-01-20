from shapely.geometry import LineString, Point
from shapely.ops import split


def split_cell(segments, idx, splitter):
    """Divide the cell at idx using either a supplied splitter geometry or direction (x|y)"""
    cells = segments.copy()
    selected = cells.loc[idx]

    # Prepare the pre-defined splitter if a direction was given
    if splitter in ["x", "y"]:
        x, y = [set(c) for c in selected.geometry.exterior.coords.xy]
        if splitter == "x":
            splitter = LineString([Point(min(x), sum(y) / 2), Point(max(x), sum(y) / 2)])
        elif splitter == "y":
            splitter = LineString([Point(sum(x) / 2, min(y)), Point(sum(x) / 2, max(y))])

    # Perform the split and update the cell dataframe while creating the necessary links between parents and children
    divided = split(selected.geometry, splitter)
    for idx_rect, rect in enumerate(divided):
        new_cell = selected.copy()
        new_idx = len(segments.index) + idx_rect
        new_cell["parent"] = idx
        new_cell["geometry"] = rect
        cells.loc[new_idx] = new_cell
        if cells.loc[idx, "children"] is None:
            cells.loc[idx, "children"] = [[new_idx]]
        else:
            cells.loc[idx, "children"].append(new_idx)

    return cells


if __name__ == "__main__":
    import geopandas as gpd
    import hvplot
    import hvplot.pandas

    cells = gpd.read_file("data/examples/braunschweig.geojson", driver="GeoJSON")
    cells = split_cell(cells, 42, "x")
    plot = cells.hvplot(
        c="class",
        geo=True,
        frame_height=1000,
        tiles="CartoDark",
        hover_cols=["floor", "ceiling"],
        alpha=0.2,
    )
    hvplot.show(plot)
