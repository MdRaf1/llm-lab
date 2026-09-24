#!/usr/bin/env python3
# Wide-active-set stressor prompt generator (seeded, reproducible).
# Goal: force near-full expert coverage on Qwen3-30B-A3B (128 experts/layer,
# top-8). High-entropy, multi-domain token stream so nearly every token routes
# to a different top-8 -> the touched expert working set approaches the whole
# 18.55 GB model, exceeding the ~13 GB reclaimable cache. If decode then
# saturates the drive, streaming is DORMANT (not dead) on this box.
import random, sys
random.seed(1729)

# Multi-domain vocabulary pools (distinct routing pressure across registers).
sci = "entropy photosynthesis mitochondria quark tectonic enzyme catalyst isotope neutrino galaxy synapse ribosome plasma fusion antibody genome protein voltage refraction covalent".split()
hist = "revolution empire dynasty treaty renaissance feudal parliament colonial armistice republic monarchy crusade suffrage reformation guillotine chancellor magna carta senate emperor".split()
tech = "kernel pointer mutex latency throughput cache register pipeline compiler heap stack syscall socket protocol daemon buffer overflow recursion hashmap thread".split()
words = "azimuth quixotic verdant lugubrious ephemeral obsidian cascade tempest nebula fjord meander sonorous brackish gossamer zephyr lattice cinnabar vermillion halcyon petrichor".split()
foreign = "بيت חבר κόσμος दुनिया 世界 mundo Welt monde мир 世界 空 река montagne río".split()
code = ["def f(x): return x*x", "for i in range(n): s+=a[i]", "SELECT id FROM t WHERE x>3", "if (p != NULL) free(p);", "x = [i**2 for i in v]", "async fn run() -> Result<()>"]

parts = []
nparts = int(sys.argv[1]) if len(sys.argv) > 1 else 900
for _ in range(nparts):
    r = random.random()
    if r < 0.35:
        pool = random.choice([sci, hist, tech, words])
        parts.append(" ".join(random.sample(pool, k=random.randint(3, 6))))
    elif r < 0.5:
        parts.append(str(random.randint(1000, 9_999_999)))
    elif r < 0.62:
        parts.append(random.choice(code))
    elif r < 0.72:
        parts.append(" ".join(random.sample(foreign, k=random.randint(2, 4))))
    else:
        subj = random.choice(sci + hist + tech)
        verb = random.choice(["governs", "precipitates", "entangles", "supersedes", "amplifies", "constrains"])
        obj = random.choice(words + sci)
        parts.append(f"The {subj} {verb} the {obj} across {random.randint(2,99)} regimes.")

text = " ".join(parts)
sys.stdout.write(text)
sys.stderr.write(f"chars={len(text)} approx_tokens={len(text)//4}\n")
