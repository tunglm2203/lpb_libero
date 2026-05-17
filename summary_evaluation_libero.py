import os
import json
import numpy as np
import argparse
from termcolor import cprint


def main(args):
    all_tasks = [
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
    ]

    all_scores_across_seeds = []
    for seed in [1, 2, 3]:
        try:
            all_scores = []
            n_tests_per_task = []
            n_task_done = 0
            for task in all_tasks:
                full_path = os.path.join(args.root, args.exp_name, task, f"eval_results_{seed}.json")
                with open(full_path, 'r') as f:
                    results = json.load(f)

                mean_score = None
                for key in list(results.keys()):
                    if key.endswith('mean_score'):
                        mean_score = results[key]
                        break
                all_scores.append(mean_score)
                n_tests_per_task.append(len(results) - 1)  # -1 for mean_score
                n_task_done += 1
            all_scores_across_seeds.append(np.mean(all_scores) * 100)
        except:
            print(f"Seed is not evaluated: {seed}")

        # cprint(f"Exp: {args.exp_name}: mean_score={np.mean(all_scores) * 100:.2f} "
        #        f"(n_test={int(np.mean(n_tests_per_task))}, n_task_done={n_task_done})", "yellow", attrs=["bold"])
    cprint(f"Exp: {args.exp_name}: mean_score={np.mean(all_scores_across_seeds):.2f}, std={np.std(all_scores_across_seeds):.2f} "
           f"(n_seeds={len(all_scores_across_seeds)})", "green", attrs=["bold"])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()
    main(args)