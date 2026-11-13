from __future__ import annotations
import random

def select_roster(settings, participants, available):
    available=sorted(set(available))
    if not available:
        raise ValueError('El motor no devolvió personajes disponibles')
    pool=list(settings.kart_pool) if settings.kart_pool is not None else available
    unknown=set(pool)-set(available)
    if unknown:
        raise ValueError(f'kart_pool contiene identificadores desconocidos: {sorted(unknown)}')
    by_slot={p.slot:p for p in participants}
    specs=[]
    for slot in range(settings.num_karts):
        p=by_slot.get(slot)
        choice=p.kart if p and p.kart else (settings.kart if p and p.kind=='external' else settings.bot_kart)
        if choice not in ('random','same_as_player') and choice not in available:
            raise ValueError(f'Kart desconocido: {choice}; ejecuta assets')
        if p and p.kind=='external' and choice=='same_as_player':
            raise ValueError('same_as_player solo es válido para bots')
        specs.append(choice)
    chosen={i:k for i,k in enumerate(specs) if k not in ('random','same_as_player')}
    used=set(chosen.values())
    rng=random.Random(settings.resolved_seed)
    # Pick observed/external karts first, then bots; fixed choices are reserved.
    order=[p.slot for p in sorted(participants,key=lambda p:p.slot) if p.kind=='external']
    order += [i for i in range(settings.num_karts) if i not in order]
    for i in order:
        if specs[i]=='random':
            candidates=[k for k in pool if k not in used] if settings.avoid_duplicate_karts else pool
            chosen[i]=rng.choice(candidates or pool)
            used.add(chosen[i])
    reference=next((chosen[p.slot] for p in sorted(participants,key=lambda p:p.slot) if p.kind=='external'),None)
    if reference is None:
        reference=settings.kart if settings.kart!='random' else rng.choice(pool)
        if reference not in available:
            raise ValueError(f'Kart desconocido: {reference}')
    for i,k in enumerate(specs):
        if k=='same_as_player': chosen[i]=reference
    return [{'slot':i,'participant':by_slot[i].name if i in by_slot else None,
             'controller':'external' if i in by_slot and by_slot[i].kind=='external' else 'stk_builtin_ai',
             'kart':chosen[i], 'observed':bool(i in by_slot and by_slot[i].observe)}
            for i in range(settings.num_karts)]
