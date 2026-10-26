"""The best rule-based agent I made.
by Bram Grooten

Adapted to reason correctly under standard CARD_KNOWLEDGE mode, where an
agent cannot directly observe its own cards' true color/rank and must
instead reason over the full set of (color, rank) combinations still
consistent with hints received so far -- including negative inference from
hints that did NOT apply to a given card. This relies on the
color_plausible/rank_plausible fields added to card_knowledge dicts in
rl_env_adjusted.py.

A card is only treated as "definitely playable" or "definitely useless" if
that holds for EVERY remaining plausible (color, rank) combination -- the
only sound notion of certainty under partial information.

Also adds a genuine hint-giving strategy: the original agent's hint choice
(legals_HLE[-1], an arbitrary "last legal hint") was never actually smart,
because under CHEAT mode an agent's own play/discard decisions never
depended on hints received anyway, so hint content was functionally
irrelevant. Under real partial-information rules, hints are the only way a
partner gains information, so find_best_hint() simulates the effect of each
possible hint on the partner's own plausibility state, and looks for one
that would make one of their currently-playable cards provably playable to
them.
"""
import random
from hanabi_learning_environment.rl_env import Agent


COLORS = ['R', 'Y', 'G', 'W', 'B']
RANKS = [0, 1, 2, 3, 4]


def is_fully_known(card):
    """True if both color and rank are directly known (hinted)."""
    return card['color'] is not None and card['rank'] is not None


def plausible_combos(card):
    """Returns the list of (color, rank) tuples still consistent with all
    hints received about this card. For a fully-known card, returns just
    the single known combo. Requires color_plausible/rank_plausible fields
    on the card dict (added in rl_env_adjusted.py); falls back to treating
    everything as unknown if they're missing (e.g. under CHEAT mode, where
    color()/rank() are always directly known anyway).
    """
    if is_fully_known(card):
        return [(card['color'], card['rank'])]
    if 'color_plausible' not in card or 'rank_plausible' not in card:
        return []
    combos = []
    for color in COLORS:
        if not card['color_plausible'][color]:
            continue
        for rank in RANKS:
            if not card['rank_plausible'][rank]:
                continue
            combos.append((color, rank))
    return combos


def best_guess(card):
    """A single representative (color, rank) guess for a card, used only
    for heuristic tie-breaking (e.g. choosing which card to discard when no
    card is provably safe either way). Picks the first plausible color and
    first plausible rank -- a reasonable deterministic placeholder for
    comparison purposes, not a probability-weighted guess.
    """
    combos = plausible_combos(card)
    return combos[0] if combos else (card['color'], card['rank'])


def playable_card(card, fireworks):
    """A card is DEFINITELY playable only if every (color, rank) combination
    still consistent with hints received would be playable. If even one
    plausible interpretation is not playable, playing risks a life token,
    so we don't call it playable.
    """
    combos = plausible_combos(card)
    if not combos:
        return False
    for color, rank in combos:
        if rank != fireworks[color]:
            return False
    return True


def _useless_single(color, rank, fireworks, discard_pile):
    """Original single-card uselessness check (Bram Grooten's logic),
    applied to one concrete, known (color, rank) combination.
    """
    if rank < fireworks[color]:
        return True  # definitely useless when firework is bigger
    elif rank == fireworks[color]:
        return False  # not useless: it's playable!
    else:
        num_discards_per_rank = [0] * 5
        for discarded in discard_pile:
            if discarded['color'] == color:
                num_discards_per_rank[discarded['rank']] += 1
        for lower_rank in range(rank):
            if lower_rank == 0:
                if num_discards_per_rank[lower_rank] >= 3:
                    return True
            elif lower_rank <= 3:
                if num_discards_per_rank[lower_rank] >= 2:
                    return True
    return False


def useless_card(card, fireworks, discard_pile):
    """A card is DEFINITELY useless only if every (color, rank) combination
    still consistent with hints received would be useless.
    """
    combos = plausible_combos(card)
    if not combos:
        return False
    for color, rank in combos:
        if not _useless_single(color, rank, fireworks, discard_pile):
            return False
    return True


def find_best_discard_option(cards, fireworks, discard_pile):
    """This function assumes that none of your cards are provably playable
    and none are provably useless, and hinting is illegal -- you must
    discard, so we minimize the damage. Uses a best-guess (color, rank) for
    each card (see best_guess): this was already a heuristic tie-breaker in
    the original design even under full information, so using a single
    representative guess rather than full plausible-set reasoning is an
    acceptable simplification for this specific fallback case.
    """
    guesses = [best_guess(card) for card in cards]

    dup_discards = []
    for idx, (color, rank) in enumerate(guesses):
        matches = sum(1 for c2, r2 in guesses if c2 == color and r2 == rank)
        if matches > 1:
            dup_discards.append(idx)
    if dup_discards:
        return random.choice(dup_discards)

    may_discard = [True] * len(cards)
    for idx, (color, rank) in enumerate(guesses):
        if rank >= 4:
            may_discard[idx] = False

    if not any(may_discard):
        discard_idx = 0
        lowest_fire = 5
        for idx, (color, rank) in enumerate(guesses):
            if fireworks[color] < lowest_fire:
                lowest_fire = fireworks[color]
                discard_idx = idx
        return discard_idx

    for card in discard_pile:
        for idx_own, (color, rank) in enumerate(guesses):
            if card['color'] == color and card['rank'] == rank:
                may_discard[idx_own] = False

    if any(may_discard):
        highest_rank = 0
        for idx, can_discard in enumerate(may_discard):
            if can_discard and guesses[idx][1] > highest_rank:
                highest_rank = guesses[idx][1]
        possible_discards = []
        for idx, can_discard in enumerate(may_discard):
            if can_discard and guesses[idx][1] == highest_rank:
                possible_discards.append(idx)
        return random.choice(possible_discards)
    else:
        highest_rank_not_5 = 0
        for idx, (color, rank) in enumerate(guesses):
            if rank > highest_rank_not_5 and rank != 4:
                highest_rank_not_5 = rank
        possible_discards = []
        for idx, (color, rank) in enumerate(guesses):
            if rank == highest_rank_not_5:
                possible_discards.append(idx)
        return random.choice(possible_discards)


def _simulate_playable_after_color_hint(known_card, true_color, fireworks):
    """Would this card become provably playable if its color were revealed
    to be true_color (leaving rank knowledge exactly as it currently is)?
    """
    simulated = dict(known_card)
    if simulated.get('rank') is not None:
        # Rank already fully known -- combined with the new color hint,
        # the card becomes fully resolved.
        simulated['color'] = true_color
    else:
        simulated['color'] = None
        simulated['color_plausible'] = {c: (c == true_color) for c in COLORS}
    return playable_card(simulated, fireworks)


def _simulate_playable_after_rank_hint(known_card, true_rank, fireworks):
    """Would this card become provably playable if its rank were revealed
    to be true_rank (leaving color knowledge exactly as it currently is)?
    """
    simulated = dict(known_card)
    if simulated.get('color') is not None:
        simulated['rank'] = true_rank
    else:
        simulated['rank'] = None
        simulated['rank_plausible'] = {r: (r == true_rank) for r in RANKS}
    return playable_card(simulated, fireworks)


def find_best_hint(obs, fireworks):
    """Looks for a hint that would make one of the partner's currently
    playable-in-reality cards provably playable to them. This is the single
    most valuable kind of hint in Hanabi: it directly enables a play next
    turn. Real hands (observed_hands) are used to know what's actually true
    about the partner's cards -- this is legitimate information under
    standard Hanabi rules, since you can always see your teammates' hands,
    only not your own.

    Falls back to hinting the partner's lowest-rank card whose rank isn't
    yet directly known, as a simple "build toward future playability"
    heuristic, if no immediately-resolving hint is found.

    Returns an HLE action dict, or None if hinting isn't useful at all
    (caller should fall back to something else, e.g. a legal hint or a
    forced discard).
    """
    partner_true_hand = obs['observed_hands'][1]
    partner_knowledge = obs['card_knowledge'][1]

    for true_card, known_card in zip(partner_true_hand, partner_knowledge):
        true_color = true_card['color']
        true_rank = true_card['rank']
        if true_rank != fireworks[true_color]:
            continue  # not currently playable in reality

        if playable_card(known_card, fireworks):
            continue  # already provably playable to them

        if _simulate_playable_after_color_hint(known_card, true_color, fireworks):
            return {'action_type': 'REVEAL_COLOR', 'target_offset': 1, 'color': true_color}

        if _simulate_playable_after_rank_hint(known_card, true_rank, fireworks):
            return {'action_type': 'REVEAL_RANK', 'target_offset': 1, 'rank': true_rank}

    # No hint directly enables a play. Fall back to building information:
    # hint the rank of the partner's lowest-rank card whose rank isn't yet
    # directly known, so it can become playable sooner as fireworks progress.
    candidates = [
        (true_card['rank'], true_card)
        for true_card, known_card in zip(partner_true_hand, partner_knowledge)
        if known_card['rank'] is None
    ]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, chosen = candidates[0]
        return {'action_type': 'REVEAL_RANK', 'target_offset': 1, 'rank': chosen['rank']}

    return None


class BramsBestRuleBasedAgent(Agent):
    """Rule-based agent, adapted to reason correctly under standard
    partial-information (CARD_KNOWLEDGE) Hanabi rules using plausibility
    tracking and a genuine hint-giving strategy, in addition to its
    original cheat-mode behavior.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.max_information_tokens = config.get('information_tokens', 8)
        self.max_life_tokens = config.get('life_tokens', 3)

    def reset(self, config=None):
        pass

    def act(self, obs):
        """Chooses the best action to play.
        Args:
            obs: dict, the HLE observation for current player
        Returns:
            dict, an HLE action
        """
        if obs['current_player_offset'] != 0:
            return None

        my_cards = obs['card_knowledge'][0]

        playable_cards = []
        fireworks = obs['fireworks']
        for idx, card in enumerate(my_cards):
            if playable_card(card, fireworks):
                playable_cards.append((idx, card))

        if playable_cards:
            ranks = [best_guess(c)[1] for _, c in playable_cards]
            lowest_rank = min(ranks)
            highest_rank = max(ranks)
            if highest_rank == 4:
                indices_4 = []
                for (idx, card), rank in zip(playable_cards, ranks):
                    if rank == 4:
                        indices_4.append(idx)
                return {'action_type': 'PLAY', 'card_index': random.choice(indices_4)}
            else:
                indices_lowest = []
                for (idx, card), rank in zip(playable_cards, ranks):
                    if rank == lowest_rank:
                        indices_lowest.append(idx)
            return {'action_type': 'PLAY', 'card_index': random.choice(indices_lowest)}

        discarding_legal = (obs['information_tokens'] < self.max_information_tokens)
        hint_possible = (obs['information_tokens'] > 0)

        if hint_possible:
            hint_action = find_best_hint(obs, fireworks)
            if hint_action is not None:
                return hint_action

        if discarding_legal:
            discard_pile = obs['discard_pile']
            useless_indices = []
            for idx, card in enumerate(my_cards):
                if useless_card(card, fireworks, discard_pile):
                    useless_indices.append(idx)

            if useless_indices:
                discard_idx = random.choice(useless_indices)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

            if hint_possible:
                legals_HLE = obs['legal_moves']
                return legals_HLE[-1]
            else:
                discard_idx = find_best_discard_option(my_cards, fireworks, discard_pile)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

        else:
            legals_HLE = obs['legal_moves']
            return legals_HLE[-1]
