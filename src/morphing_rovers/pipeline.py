"""End-to-end optimisation: rover masks from terrain clustering, then the controller by imitation learning."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from morphing_rovers.chromosome import (
    SWITCH_BIAS,
    SWITCH_WEIGHTS,
    always_switch,
    assemble,
    load_chromosome,
    save_chromosome,
    split,
)
from morphing_rovers.config import Config
from morphing_rovers.experts import Expert, make_expert
from morphing_rovers.imitation import train_controller
from morphing_rovers.masks import build_masks
from morphing_rovers.routes import SwitchOracle, refine_masks, search_experts
from morphing_rovers.submission import write_submission
from morphing_rovers.udp import morphing_rover_UDP

logger = logging.getLogger(__name__)


def run(cfg: Config) -> np.ndarray:
    """Optimise masks for planned routes, train the controller, and write the submission file."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    masks = build_masks(cfg.masks, udp.env, cfg.seed, output_dir)
    init = cfg.imitation.controller_init
    _, network = split(udp.example() if init == "example" else load_chromosome(init))
    baseline = always_switch(assemble(masks, network))
    baseline_fitness = float(udp.fitness(baseline)[0])
    logger.info("Always-switch baseline fitness %.4f over the 30 scenarios", baseline_fitness)
    if init != "example":
        supplied = load_chromosome(init)
        supplied_fitness = float(udp.fitness(supplied)[0])
        if supplied_fitness < baseline_fitness:
            baseline, baseline_fitness = supplied, supplied_fitness
            logger.info("Using supplied chromosome as incumbent (fitness %.4f)", baseline_fitness)

    scenarios = range(30) if cfg.imitation.scenarios is None else cfg.imitation.scenarios
    base_experts = {
        scenario: make_expert(scenario, cfg.imitation.radius_factors.get(scenario),
                              cfg.imitation.arc_points, cfg.imitation.lookahead)
        for scenario in scenarios
    }

    # Alternate between selecting a useful heading offset, tracing that route, and fitting the masks to
    # the modes that the cooldown-aware schedule actually uses there.
    route_choices: dict[int, dict[str, float | None]] = {}
    for round_id in range(cfg.schedule.rounds):
        experts, routes, route_choices = search_experts(
            udp, masks, base_experts, cfg.schedule.radius_factors,
            cfg.schedule.offset_candidates_degrees, cfg.imitation.arc_points,
        )
        masks = refine_masks(masks, routes, replace(cfg.schedule, rounds=1))
        np.save(output_dir / "masks.npy", masks)
        logger.info("Route and mask refinement round %d/%d complete", round_id + 1, cfg.schedule.rounds)

    experts, routes, route_choices = search_experts(
        udp, masks, base_experts, cfg.schedule.radius_factors,
        cfg.schedule.offset_candidates_degrees, cfg.imitation.arc_points,
    )
    with open(output_dir / "route_choices.json", "w") as f:
        json.dump(route_choices, f, indent=2)
    np.save(output_dir / "masks.npy", masks)
    switch_oracles = {
        scenario: (route, SwitchOracle(route, route.velocities(masks)))
        for scenario, route in routes.items()
    }

    chromosome = always_switch(assemble(masks, network))
    # Start the switch head neutral so imitation can learn when a scheduled morph is worthwhile.
    chromosome[SWITCH_WEIGHTS] = 0.0
    chromosome[SWITCH_BIAS] = 0.0
    best, _ = train_controller(udp, chromosome, cfg.imitation, output_dir, experts, switch_oracles)
    trained_fitness = float(udp.fitness(best)[0])
    if baseline_fitness <= trained_fitness:
        logger.info("Keeping the safer always-switch baseline (%.4f) over trained controller (%.4f)",
                    baseline_fitness, trained_fitness)
        best = baseline

    np.save(output_dir / "masks.npy", split(best)[0])
    save_chromosome(output_dir / "best_chromosome.npy", best)
    (fitness,), _ = udp.pretty(best)
    write_submission(best, output_dir / "submission.json", cfg.submission.name, cfg.submission.description)
    logger.info("Best chromosome (fitness %.4f over the 30 scenarios) saved in %s", fitness, output_dir)
    return best


def _with_offsets(experts: dict[int, Expert], offsets: dict[int, float]) -> dict[int, Expert]:
    """Copy experts with their scenario-specific heading corrections."""
    return {scenario: Expert(expert.waypoints, expert.lookahead, offsets.get(scenario, 0.0))
            for scenario, expert in experts.items()}


def sweep(cfg: Config) -> dict[int, dict[str, float]]:
    """Imitation learning on single scenarios with each candidate arc radius factor, to choose
    `imitation.radius_factors`. Results are written to sweep.json."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    chromosome = initial_chromosome(cfg, udp, output_dir)

    results: dict[int, dict[str, float]] = {}
    best_factors = {}
    for scenario in cfg.sweep.scenarios:
        results[scenario] = {}
        for factor in cfg.sweep.radius_factors:
            imitation = replace(cfg.imitation, scenarios=[scenario], radius_factors={scenario: factor})
            _, fitness = train_controller(udp, chromosome, imitation)
            results[scenario]["straight" if factor is None else str(factor)] = fitness
            logger.info("scenario %d, radius factor %s: best fitness %.4f", scenario, factor, fitness)
        best_factors[scenario] = cfg.sweep.radius_factors[int(np.argmin(list(results[scenario].values())))]
        with open(output_dir / "sweep.json", "w") as f:
            json.dump(results, f, indent=2)
    logger.info("Best radius factor per scenario (None: straight line): %s", best_factors)
    return results


def initial_chromosome(cfg: Config, udp: morphing_rover_UDP, output_dir: Path) -> np.ndarray:
    """Optimised masks combined with the initial controller, which always switches to the best mode."""
    masks = build_masks(cfg.masks, udp.env, cfg.seed, output_dir)
    init = cfg.imitation.controller_init
    _, network = split(udp.example() if init == "example" else load_chromosome(init))
    chromosome = always_switch(assemble(masks, network))
    np.save(output_dir / "masks.npy", split(chromosome)[0])
    return chromosome


def _prepare(cfg: Config) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    torch.manual_seed(cfg.seed)
    return output_dir
