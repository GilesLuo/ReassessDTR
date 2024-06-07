import torch
from HD4RL.core.helper_fn import get_class, policy_load
import os
from HD4RL.utils.misc import to_bool
from DTRGym import buffer_registry
from HD4RL.utils.summary import get_sweep, get_best_run
import importlib
import wandb
import re
import warnings
from HD4RL.utils.network import CalibratedNet
from HD4RL.utils.data import TianshouDataset, collate_batch_seq2seq, load_buffer
from HD4RL.offline.offlineRLHparams import DummyHparam

warnings.filterwarnings("ignore")


def get_behavioural_fn(project, env_name, algo_name, device, behavioural_model_path=None):
    # try:
    behavioural_run = get_sweep(project, f"{env_name}-all_ope_train-",
                                algo_name)
    behavioural_run = get_best_run(behavioural_run, "val/all_val-PatientWiseF1", maximize=True)
    behavioural_fn = get_class("obj", algo_name, offline=True)(
        env_name + "-discrete",
        DummyHparam(), device=device,
        logger=None,
        train_buffer_name=None, val_buffer_name=None,
        test_buffer_keyword=None,
        OPE_methods=None,  # placeholder
        metric=None,  # placeholder
        test_online=False).define_policy(**behavioural_run.to_dict())
    behavioural_model_path = os.path.join(behavioural_run["model_dir"], "policy.pth") \
        if behavioural_model_path is None else behavioural_model_path
    behavioural_fn = policy_load(behavioural_fn, behavioural_model_path,
                                 device, is_train=False)
    print("Successfully load behavioural function from discrete-imitation-rnn checkpoint")
    behavioural_args = {"model": behavioural_fn,
                        "stack_num": behavioural_run["cat_num"] * behavioural_run["stack_num"]}
    # except Exception as e:
    #     print("Failed to get wandb sweep. Did you experiment behavioural function beforehand?"
    #           " Error: {}. Continue with no behavioural function".format(e))
    #     behavioural_args = {"model": None}
    return behavioural_args


def get_value_fn(project, env_name, algo_name, device, value_model_path=None):
    try:
        value_run = get_sweep(project, f"{env_name}-all_ope_train-",
                              algo_name)
        value_run = get_best_run(value_run, "val/all_val-TD", maximize=True)
        value_fn = get_class("obj", algo_name, offline=True)(
            f"{env_name}-discrete",
            DummyHparam(), device=device,
            logger=None,
            train_buffer_name=None, val_buffer_name=None,
            test_buffer_keyword=None,
            OPE_methods=None,  # placeholder
            metric=None,  # placeholder
            test_online=False).define_policy(**value_run.to_dict())
        value_model_path = os.path.join(value_run["model_dir"], "policy.pth") \
            if value_model_path is None else value_model_path
        value_fn = policy_load(value_fn, value_model_path, device, is_train=False)
        print("Successfully load value function from offline sarsa checkpoint")
        value_args = {"model": value_fn, "stack_num": value_run["cat_num"] * value_run["stack_num"]}
    except Exception as e:
        print("Failed to get wandb sweep. Did you experiment value function beforehand?"
              " Error: {}. Continue with no value function".format(e))
        value_args = {"model": None}
    return value_args


def load_behavioural_fn(project, env_name, algo_name, device, behavioural_model_path,
                        calibrate=False, calibrated_model_path=None, val_buffer=None):
    bc_args = get_behavioural_fn(project, env_name, algo_name, device, behavioural_model_path)
    if not calibrate:
        print("No calibration is performed, return behavioural function directly")
        return bc_args
    if bc_args["model"] is None:
        raise ValueError("No behavioural function found, cannot calibrate")
    if [calibrated_model_path, val_buffer].count(None) > 0:
        raise ValueError("calibrated_model_path and val_buffer must be both specified when calibrate is True")
    if os.path.exists(calibrated_model_path):
        print("Calibrated behavioural function found in {}, load directly".format(calibrated_model_path))
        bc_args["model"].model = torch.load(calibrated_model_path)
    else:
        print(
            "Calibrated behavioural function not found in {}, experiment calibrating...".format(calibrated_model_path))
        model = bc_args["model"].model
        dataset = TianshouDataset(val_buffer, stack_num=bc_args["stack_num"])
        calibrated_model = CalibratedNet(model).grid_search_calibration(dataset)
        os.makedirs(os.path.dirname(calibrated_model_path), exist_ok=True)
        torch.save(calibrated_model, calibrated_model_path)
        print("Calibrated model saved to {}".format(calibrated_model_path))
        bc_args["model"].model = calibrated_model
    return bc_args


#         check if the calibrated model exists


def call_agent():
    try:
        obj = obj_class(f"{args.env}-{get_class('type', args.algo_name, True)}", hparam_space, device=args.device,
                        logger="wandb",
                        train_buffer_name=args.train_buffer, val_buffer_name=args.val_buffer,
                        test_buffer_keyword=args.test_buffer_keyword,
                        OPE_methods=args.OPE_methods,
                        OPE_args=ope_args,
                        metric=args.OPE_metric,
                        test_online=args.test_online)
        obj.search_once(None, goal_name)
    except TimeoutError as e:
        # Update the status to 'crashed' due to timeout
        wandb.run.summary["status"] = "crashed"
        wandb.run.summary["failure_reason"] = str(e)
        wandb.run.finish()
        raise e
    else:
        # Finish the wandb experiment normally if no issues
        wandb.finish()
    return


if __name__ == "__main__":
    import argparse

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # training-aid hyperparameters
    parser.add_argument("--project", type=str, default="SepsisRL-ICML")
    parser.add_argument("--role", type=str, default="run_single",
                        choices=["sweep", "agent", "run_single"],
                        help="sweep is to experiment a sweep controller on the w&b server."
                             "agent is to experiment a single experiment controlled by a created sweep."
                             "run_single is to experiment a single experiment without sweep.")

    parser.add_argument("--sweep_id", type=str, default="mzdwn9fq")
    parser.add_argument("--mode", type=str, choices=["val-online-test-online",
                                                     "val-offline-test-offline",
                                                     "val-offline-test-online",
                                                     "val-offline-test-offline&online"],
                        default="val-offline-test-offline")
    parser.add_argument("--env", type=str, default="MIMIC3SepsisNEWS2Env")
    parser.add_argument("--logdir", type=str, default="./debugging_log")

    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--test_num", type=int, default=100,
                        help="number of test envs. Not useful if test_online=False")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument('--linear', dest='linear', action='store_true')
    parser.add_argument('--nonlinear', dest='linear', action='store_false')
    parser.set_defaults(linear=False)
    parser.add_argument('--all_soften', default=False, action='store_true', )
    parser.add_argument('--calibrate_behavioural', default=False, action='store_true', )
    parser.add_argument('--bc_algo', default="discrete-imitation-rnn")
    parser.add_argument('--behavioural_model_path',
                        default="/home/reub0014/projects/SimMedEnv/experiment/saved_models/bc_policy.pth", )
    parser.add_argument('--calibrated_model_path', default="/home/reub0014/projects/"
                                                           "SimMedEnv/experiment/saved_models/calibrated_model.pt")

    parser.add_argument('--value_algo', default="offlinesarsa-rnn")
    parser.add_argument('--value_model_path',
                        default="/home/reub0014/projects/SimMedEnv/experiment/saved_models/NEWS2_value_policy.pth")
    parser.add_argument("--train_buffer", type=str, default="all_train")
    parser.add_argument("--val_buffer", type=str, default="all_val")
    parser.add_argument("--test_buffer_keyword", type=str, default="all_test", help="keyword to find all test buffer")
    parser.add_argument("--test_online", type=to_bool, default=False)
    parser.add_argument("--OPE_methods", nargs='+', choices=['WIS', 'WIS_bootstrap',
                                                             'WIS_truncated', 'WIS_bootstrap_truncated',
                                                             'PatientWiseF1', "SampleWiseF1", "TD",
                                                             "DR", "doseRMSE"],
                        default=["doseRMSE"],
                        # default=["TD"],
                        help="Select one or more options from the list")
    parser.add_argument("--OPE_metric", type=str, default="doseRMSE")
    parser.add_argument("--goal", type=str, default="maximize", choices=["maximize", "minimize"], )
    parser.add_argument("--algo_name", type=str, default="discrete-iql-obs_cat", choices=["dqn",
                                                                                          "dqn-obs_cat",

                                                                                          "discrete-bcq",
                                                                                          "discrete-bcq-obs_cat",

                                                                                          "discrete-cql",
                                                                                          "discrete-cql-obs_cat",

                                                                                          "discrete-iql",
                                                                                          "discrete-iql-obs_cat",
                                                                                          # OPE models below
                                                                                          "discrete-imitation",
                                                                                          "discrete-imitation-rnn",
                                                                                          "discrete-imitation-obs_cat",

                                                                                          "offlinesarsa",
                                                                                          "offlinesarsa-rnn",
                                                                                          "offlinesarsa-obs_cat",

                                                                                          ])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_known_args()[0]
    args.seed = int(args.seed) if args.seed not in [None, "None", "none", "NONE"] else None
    hparam_class = get_class("hparam", args.algo_name, offline=True)
    obj_class = get_class("obj", args.algo_name, offline=True)
    study_name = f"{args.env}-{args.train_buffer}-{args.algo_name}"
    os.makedirs(args.logdir, exist_ok=True)
    module = importlib.import_module(f"DTRGym.{args.env}")

    hparam_space = hparam_class(args.algo_name,
                                args.logdir,
                                args.seed,
                                args.test_num,  # number of test envs
                                args.epoch,
                                None,
                                # number of training steps per epoch, if None, it is the same as the length of the buffer
                                linear=args.linear,
                                all_soften=args.all_soften, )
    search_space = hparam_space.get_search_space()
    whole_config = {**{k: {"value": v} for k, v in hparam_space.get_meta_params().items()}, **search_space}
    goal_name = "test/all_test-" + args.OPE_metric if args.test_buffer_keyword in ["test", "all_test"] \
        else "test/all_val-" + args.OPE_metric  # adapt to ope model training

    ope_args = {}
    # get value function from offline sarsa checkpoint
    if "IS" in args.OPE_methods or "WIS" in args.OPE_methods or "WIS_bootstrap" in args.OPE_methods or \
            "WIS_truncated" in args.OPE_methods or "WIS_bootstrap_truncated" in args.OPE_methods or \
            "DR" in args.OPE_methods or "WDR" in args.OPE_methods or "PDDR" in args.OPE_methods or "PDWDR" in args.OPE_methods:
        ope_args["behavioural_fn"] = load_behavioural_fn(args.project, "MIMIC3SepsisNEWS2Env",
                                                         args.bc_algo, args.device,
                                                         behavioural_model_path=args.behavioural_model_path,
                                                         calibrate=args.calibrate_behavioural,
                                                         calibrated_model_path=args.calibrated_model_path,
                                                         val_buffer=load_buffer(
                                                             buffer_registry.make(args.env, "all_test")))  # placeholder

    if "DR" in args.OPE_methods or "WDR" in args.OPE_methods or "PDDR" in args.OPE_methods or "PDWDR" in args.OPE_methods:
        ope_args["value_fn"] = get_value_fn(args.project, args.env, args.value_algo, args.device,
                                            value_model_path=args.value_model_path)

    print("All prepared. Start to experiment")
    if args.role == "sweep":
        sweep_configuration = {
            "method": "grid",
            "name": study_name,
            "metric": {"goal": args.goal, "name": goal_name},
            "parameters": whole_config
        }
        sweep_id = wandb.sweep(sweep_configuration, project=args.project)
        wandb.agent(sweep_id=sweep_id, function=call_agent, project=args.project, entity="gilesluo")
    else:
        if args.role == "agent":
            wandb.agent(sweep_id=args.sweep_id, function=call_agent, project=args.project, entity="gilesluo")
        if args.role == "run_single":
            obj = obj_class(f"{args.env}-{get_class('type', args.algo_name, True)}", hparam_space, device=args.device,
                            logger="wandb",
                            train_buffer_name=args.train_buffer, val_buffer_name=args.val_buffer,
                            test_buffer_keyword=args.test_buffer_keyword,
                            OPE_methods=args.OPE_methods,
                            OPE_args=ope_args,
                            metric=args.OPE_metric,
                            test_online=args.test_online)
            config_dict = hparam_space.sample(mode="random")
            config_dict.update({k: v for k, v in hparam_space.get_meta_params().items()})
            config_dict["env_name"] = f"{args.env}"
            obj.search_once(config_dict, goal_name)
        else:
            print("role must be one of [sweep, agent, run_single], get {}".format(args.role))
            raise NotImplementedError
