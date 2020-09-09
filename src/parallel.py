from tqdm import tqdm

from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessPool as Pool

# from pathos.threading import ThreadPool as Pool
from collections.abc import Sized


def _parallel(ordered, function, *iterables, **kwargs):
    """Returns a generator for a parallel map with a progress bar.
    Arguments:
        ordered(bool): True for an ordered map, false for an unordered map.
        function(Callable): The function to apply to each element of the given Iterables.
        iterables(Tuple[Iterable]): One or more Iterables containing the data to be mapped.
    Returns:
        A generator which will apply the function to each element of the given Iterables
        in parallel in order with a progress bar.
    """

    # Extract num_cpus
    num_cpus = kwargs.pop("num_cpus", None)
    do_tqdm = kwargs.pop("do_tqdm", True)

    # Determine num_cpus
    if num_cpus is None:
        num_cpus = cpu_count()
    elif type(num_cpus) == float:
        num_cpus = int(round(num_cpus * cpu_count()))

    # Determine length of tqdm (equal to length of shortest iterable)
    length = min(len(iterable) for iterable in iterables if isinstance(iterable, Sized))

    # Create parallel generator
    map_type = "imap" if ordered else "uimap"
    pool = Pool(num_cpus)
    map_func = getattr(pool, map_type)

    # create iterable
    items = map_func(function, *iterables)

    # add progress bar
    if do_tqdm:
        items = tqdm(items, total=length, **kwargs)

    for item in items:
        yield item

    pool.clear()


def p_umap(function, *iterables, **kwargs):
    """Performs a parallel unordered map with a progress bar."""

    ordered = False
    generator = _parallel(ordered, function, *iterables, **kwargs)
    result = list(generator)

    return result


def p_map(function, *iterables, **kwargs):
    """Performs a parallel ordered map with a progress bar."""

    ordered = True
    generator = _parallel(ordered, function, *iterables, **kwargs)
    result = list(generator)

    return result
