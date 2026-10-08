from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    manipulated_objects: tuple[str, ...]
    relevant_entities: tuple[str, ...]
    articulation_joints: tuple[str, ...]
    phases: tuple[str, ...]


TASK_SPECS: dict[int, TaskSpec] = {
    0: TaskSpec(
        task_id=0,
        manipulated_objects=("alphabet_soup_1", "tomato_sauce_1"),
        relevant_entities=("alphabet_soup_1", "tomato_sauce_1", "basket_1"),
        articulation_joints=(),
        phases=("approach_object_1", "grasp_object_1", "place_in_basket_1", "approach_object_2", "grasp_object_2", "place_in_basket_2"),
    ),
    2: TaskSpec(
        task_id=2,
        manipulated_objects=("moka_pot_1",),
        relevant_entities=("moka_pot_1", "flat_stove_1"),
        articulation_joints=("flat_stove_1_button",),
        phases=("turn_knob", "approach_pot", "grasp_pot", "transport_pot", "place_on_stove"),
    ),
    3: TaskSpec(
        task_id=3,
        manipulated_objects=("akita_black_bowl_1",),
        relevant_entities=("akita_black_bowl_1", "white_cabinet_1"),
        articulation_joints=("white_cabinet_1_bottom_level",),
        phases=("open_drawer", "approach_bowl", "grasp_bowl", "transport_bowl", "place_in_drawer", "close_drawer"),
    ),
    5: TaskSpec(
        task_id=5,
        manipulated_objects=("black_book_1",),
        relevant_entities=("black_book_1", "desk_caddy_1"),
        articulation_joints=(),
        phases=("approach_book", "grasp_book", "transport_book", "place_in_caddy"),
    ),
    9: TaskSpec(
        task_id=9,
        manipulated_objects=("white_yellow_mug_1",),
        relevant_entities=("white_yellow_mug_1", "microwave_1"),
        articulation_joints=("microwave_1_microjoint",),
        phases=("approach_mug", "grasp_mug", "transport_mug", "place_in_microwave", "close_door"),
    ),
}
