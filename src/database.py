try:
    from src.utils import utils

except ImportError:
    import utils


def get_accepted_cfgs(cfgs, used_hashes, variable, variable_subset=None, sort_by_variable=False):
    accepted_cfgs = []

    for cfg in cfgs:
        used_hashes.add(cfg.hash)
        if variable_subset is None or cfg[variable] in variable_subset:
            accepted_cfgs.append(cfg)
    if sort_by_variable:
        accepted_cfgs = sorted(accepted_cfgs, key=lambda cfg: cfg[variable])
    return accepted_cfgs, used_hashes


def passes_extra_selections_cut(query_cfg, extra_selections):
    for key in extra_selections.keys():
        if query_cfg[key] != extra_selections[key]:
            return False
    return True


def get_MCMC_data(
    variable="all", variable_subset=None, sort_by_variable=True, N_max=None, extra_selections=None
):

    db_cfg = utils.get_db_cfg()

    if variable == "all":
        return [utils.DotDict(cfg) for cfg in db_cfg]

    if N_max is None:
        N_max = len(db_cfg)

    if extra_selections is None:
        extra_selections = {}

    used_hashes = set()
    cfgs_to_plot = []
    for query_cfg in db_cfg:
        # break
        query_cfg.pop(variable, None)
        hash_ = query_cfg.pop("hash", None)
        if hash_ in used_hashes:
            continue

        if not passes_extra_selections_cut(query_cfg, extra_selections):
            continue

        cfgs = utils.query_cfg(query_cfg)
        if len(cfgs) != 1:
            # break
            accepted_cfgs, used_hashes = get_accepted_cfgs(
                cfgs,
                used_hashes,
                variable,
                variable_subset,
                sort_by_variable,
            )
            cfgs_to_plot.append(accepted_cfgs)
            if len(cfgs_to_plot) >= N_max:
                return cfgs_to_plot
    return cfgs_to_plot
