# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 16:
# 106900 Gonçalo Aleixo
# 106937 André Melo

import sys
import itertools
from collections import deque, defaultdict

# ——————————————————————————————————————————————————————————————————————————————
#  Geração de orientações (rotações + reflexões) para tetraminos
# ——————————————————————————————————————————————————————————————————————————————

def rotate_90(shape):
    return [(c, -r) for (r, c) in shape]

def reflect_horiz(shape):
    return [(r, -c) for (r, c) in shape]

def normalize(shape):
    min_r = min(r for r,c in shape)
    min_c = min(c for r,c in shape)
    return tuple(sorted((r-min_r, c-min_c) for (r,c) in shape))

def all_orientations(base_shape):
    variants = set()
    s = tuple(base_shape)
    for _ in range(4):
        s = tuple(rotate_90(list(s)))
        variants.add(normalize(s))
        variants.add(normalize(reflect_horiz(list(s))))
    return [list(offsets) for offsets in variants]

# Definições “base” dos tetraminos
L_base = [(0,0),(1,0),(2,0),(2,1)]
I_base = [(0,0),(1,0),(2,0),(3,0)]
T_base = [(0,1),(1,0),(1,1),(1,2)]
S_base = [(0,1),(0,2),(1,0),(1,1)]

L_orients = all_orientations(L_base)
I_orients = all_orientations(I_base)
T_orients = all_orientations(T_base)
S_orients = all_orientations(S_base)

ALL_SHAPES = {
    'L': L_orients,
    'I': I_orients,
    'T': T_orients,
    'S': S_orients
}

# ——————————————————————————————————————————————————————————————————————————————
#  Classificação de forma (L, I, T ou S) a partir de 4 células
# ——————————————————————————————————————————————————————————————————————————————

def classify_shape(cells):
    """
    Dados quatro pontos (r,c), devolve:
     - 'I' se todos na mesma linha OU na mesma coluna;
     - caso contrário, se algum ponto interno tiver 3 vizinhos → 'T';
     - senão, se alguma linha ou coluna tiver exatamente 3 desses pontos → 'L';
     - senão → 'S'.
    """
    rows = {r for r,c in cells}
    cols = {c for r,c in cells}
    if len(rows) == 1 or len(cols) == 1:
        return 'I'
    # calcula grau interno de cada célula
    degrees = []
    for (r,c) in cells:
        nb = 0
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            if (r+dr, c+dc) in cells:
                nb += 1
        degrees.append(nb)
    if max(degrees) == 3:
        return 'T'
    row_count = {}
    col_count = {}
    for (r,c) in cells:
        row_count[r] = row_count.get(r,0) + 1
        col_count[c] = col_count.get(c,0) + 1
    if any(v == 3 for v in row_count.values()) or any(v == 3 for v in col_count.values()):
        return 'L'
    return 'S'


# ——————————————————————————————————————————————————————————————————————————————
#  Verifica conectividade global: todas as células “pretas” formam um único bloco ortogonal
# ——————————————————————————————————————————————————————————————————————————————

def flood_fill_connectivity(final_mask, board):
    """
    final_mask: inteiro cujo binário marca as células pintadas (“pretas”).
    Retorna True se TODAS essas células estiverem exatamente em um único componente ortogonal.
    """
    if final_mask == 0:
        return False
    # pega o bit menos significativo ‘1’ para iniciar BFS
    low = final_mask & -final_mask
    start = low.bit_length() - 1
    visited = {start}
    queue = deque([start])
    total = bin(final_mask).count("1")
    while queue:
        idx = queue.popleft()
        r,c = board.bit_to_cell[idx]
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc = r+dr, c+dc
            if 0 <= nr < board.N and 0 <= nc < board.M:
                nid = board.cell_to_bit[(nr,nc)]
                if ((final_mask >> nid) & 1) and (nid not in visited):
                    visited.add(nid)
                    queue.append(nid)
    return len(visited) == total


# ——————————————————————————————————————————————————————————————————————————————
#  Classe “Board” – pré-cálculo de domínios, vizinhanças e bitboards
# ——————————————————————————————————————————————————————————————————————————————

class Board:
    def __init__(self, region_map):
        """
        region_map: lista de listas de int (cada célula contém o ID da região).
        Constrói:
         - cell_to_bit e bit_to_cell para representar cada célula num único inteiro [0..N*M-1].
         - region_cells[rid] = lista de (r,c) pertencentes à região rid.
         - neighbors[rid] = conjunto de regiões ortogonalmente adjacentes a rid.
         - region_domains[rid] = lista de todos os possíveis tetraminós (4 cels conectadas,
                               excluindo “2×2”), classificados em:
                               { 'shape','cells','mask','blocks2x2' }.
        """
        self.region_map = region_map
        self.N = len(region_map)
        self.M = len(region_map[0]) if self.N > 0 else 0

        # mapeamento (r,c) ↔ índice único
        self.cell_to_bit = {}
        self.bit_to_cell = {}
        idx = 0
        for r in range(self.N):
            for c in range(self.M):
                self.cell_to_bit[(r,c)] = idx
                self.bit_to_cell[idx] = (r,c)
                idx += 1
        self.total_cells = self.N * self.M

        # region_cells[rid] = lista de células (r,c)
        self.region_cells = defaultdict(list)
        for r in range(self.N):
            for c in range(self.M):
                rid = region_map[r][c]
                self.region_cells[rid].append((r,c))

        # vizinhanças ortogonais entre regiões
        self.neighbors = {rid:set() for rid in self.region_cells}
        for r in range(self.N):
            for c in range(self.M):
                rid = region_map[r][c]
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.N and 0 <= nc < self.M:
                        rid2 = region_map[nr][nc]
                        if rid2 != rid:
                            self.neighbors[rid].add(rid2)
                            self.neighbors[rid2].add(rid)

        # constrói domínios iniciais de cada região
        self.region_domains = {}
        self._build_all_candidates()

    def _build_all_candidates(self):
        """
        Para cada região rid:
         1) iteramos combos de 4 células;
         2) testamos conectividade interna (BFS/DFS);
         3) excluímos qualquer “O” (2×2 puro);
         4) classificamos classe = classify_shape();
         5) montamos bitmask das 4 células;
         6) pré-calculamos todas as máscaras 2×2 (subquadrados) que tocam essas 4 células.
        Armazenamos, para cada candidato, um dicionário:
          { 'shape': 'L'/'I'/'T'/'S',
            'cells': ( (r1,c1),...,(r4,c4) ),
            'mask': inteiro com exatamente 4 bits em 1,
            'blocks2x2': [cada bitmask 2×2 que intersecta essas 4 cels] }
        """
        for rid, cells in self.region_cells.items():
            if len(cells) < 4:
                self.region_domains[rid] = []
                continue

            raw_list = []
            for comb in itertools.combinations(cells, 4):
                # (2) conectividade ortogonal interna
                stack = [comb[0]]
                visited = {comb[0]}
                while stack:
                    x,y = stack.pop()
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nxt = (x+dx, y+dy)
                        if nxt in comb and nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)
                if len(visited) != 4:
                    continue

                # (3) excluir “O” (2×2)
                rows = {x for x,y in comb}
                cols = {y for x,y in comb}
                if len(rows) == 2 and len(cols) == 2:
                    continue

                # (4) classifica
                shape = classify_shape(comb)

                # (5) monta bitmask
                mask = 0
                for (x,y) in comb:
                    mask |= (1 << self.cell_to_bit[(x,y)])

                raw_list.append((shape, tuple(comb), mask))

            # (6) para cada candidato, pré-calcula blocos 2×2 que intersectam
            final_list = []
            for (shape, tupcells, mask) in raw_list:
                blocks = []
                for (x,y) in tupcells:
                    for dx in (0,-1):
                        for dy in (0,-1):
                            i0, j0 = x+dx, y+dy
                            if 0 <= i0 < self.N-1 and 0 <= j0 < self.M-1:
                                bm = 0
                                for (ii,jj) in ((i0,j0),(i0,j0+1),(i0+1,j0),(i0+1,j0+1)):
                                    bm |= (1 << self.cell_to_bit[(ii,jj)])
                                # se este 2×2 intersecta as 4 células, registra bm
                                if (bm & mask) != 0:
                                    blocks.append(bm)
                final_list.append({
                    'shape': shape,
                    'cells': tupcells,
                    'mask': mask,
                    'blocks2x2': blocks
                })

            # itertools.combinations gera em ordem lex por índice, e classify_shape
            # associa L→I→T→S implicitamente, mas a ordem não é fundamental.
            self.region_domains[rid] = final_list

    @staticmethod
    def parse_all_instances(text):
        """
        Recebe todo stdin como string, separa puzzles por linha em branco.
        Retorna lista de “region_map” (cada um é lista de listas de int).
        """
        segmentos = [seg for seg in text.strip().split('\n\n') if seg.strip()]
        puzzles = []
        for seg in segmentos:
            region_map = []
            for line in seg.splitlines():
                tokens = line.strip().split()
                if not tokens:
                    continue
                region_map.append(list(map(int, tokens)))
            puzzles.append(region_map)
        return puzzles


# ——————————————————————————————————————————————————————————————————————————————
#  Função de Resolução: Backtracking + MRV + LCV + Podas locais
# ——————————————————————————————————————————————————————————————————————————————

def solve_one_puzzle(region_map):
    board   = Board(region_map)
    regions = list(board.region_cells.keys())
    R       = len(regions)

    # --- 1) Pré-cálculo para forward-check de conectividade ---
    # Máscara de todas as células de cada região
    region_cell_mask = {}
    for r in regions:
        m = 0
        for (x,y) in board.region_cells[r]:
            m |= (1 << board.cell_to_bit[(x,y)])
        region_cell_mask[r] = m

    # free_mask = união de todas as células de regiões ainda não atribuídas
    free_mask = 0
    for r in regions:
        free_mask |= region_cell_mask[r]

    # --- Estado dinâmico da busca ---
    grid          = [[region_map[r][c] for c in range(board.M)] for r in range(board.N)]
    occupied_mask = 0
    assigned      = set()
    region_to_val = {}

    def valid_list(r):
        res = []
        for v in board.region_domains[r]:
            shape = v['shape']
            cells = v['cells']
            mask = v['mask']
            blocks = v['blocks2x2']

            # 1a) NÃO pode haver sobreposição de máscara
            if (occupied_mask & mask) != 0:
                continue

            # 1b) NÃO pode haver um bloco 2×2 inteiro preto
            #    (testamos: (occupied_mask|mask) contém algum blocks2x2 completo?)
            combined = occupied_mask | mask
            flag2x2 = False
            for bm in blocks:
                if (combined & bm) == bm:
                    flag2x2 = True
                    break
            if flag2x2:
                continue

            # 1c) NÃO pode haver adjacência ortogonal de mesma “shape”
            ok = True
            for (x,y) in cells:
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < board.N and 0 <= ny < board.M:
                        if grid[nx][ny] == shape:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue

            res.append(v)
        return res

    # —————————————————————————————————————————————————————————————
    #  2) Escolha de região: MRV dinâmico (minimo número de candidatos atualmente válidos)
    # —————————————————————————————————————————————————————————————
    def choose_region():
        best = None
        best_count = 10**9
        best_list = None
        for r in regions:
            if r in assigned:
                continue
            vl = valid_list(r)
            cnt = len(vl)
            # se um deles já ficou sem candidatos, falha imediata
            if cnt == 0:
                return r, vl
            if cnt < best_count:
                best_count = cnt
                best = r
                best_list = vl
        return best, best_list

    # —————————————————————————————————————————————————————————————
    #  3) LCV: entre os candidatos de r, ordena pelo menor “conflito local”
    #     para vizinhos não atribuídos.
    # —————————————————————————————————————————————————————————————
    def lcv_order(r, candidates):
        scores = []
        for v in candidates:
            shape_v = v['shape']
            cells_v = v['cells']
            conflict_count = 0
            # para cada vizinho s de r que ainda não foi atribuído,
            # contamos quantos candidatos u em board.region_domains[s]
            # teriam “shape_u == shape_v” e estariam ortogonalmente adjacentes
            for s in board.neighbors[r]:
                if s in assigned:
                    continue
                for u in board.region_domains[s]:
                    if u['shape'] != shape_v:
                        continue
                    # se ANY célula de v ficar ortogonalmente adjacente a alguma de u, incrementa conflito
                    conflict = False
                    for (x1,y1) in cells_v:
                        for (x2,y2) in u['cells']:
                            if abs(x1-x2) + abs(y1-y2) == 1:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        conflict_count += 1
            scores.append((conflict_count, v))
        scores.sort(key=lambda x: x[0])
        return [v for (_,v) in scores]

    sys.setrecursionlimit(10000)
    def backtrack():
        nonlocal occupied_mask, free_mask

        # 1) se todas regiões atribuídas, checa conectividade final
        if len(assigned) == R:
            return flood_fill_connectivity(occupied_mask, board)

        # 2) escolhe a próxima região por MRV
        r, vl = choose_region()
        if r is None or not vl:
            return False
        assigned.add(r)

        # 3) ordena candidatos por LCV
        ordered = lcv_order(r, vl)

        for v in ordered:
            shape, cells, mask, blocks = v['shape'], v['cells'], v['mask'], v['blocks2x2']

            # 4) testes básicos: sobreposição, bloco 2×2 e adjacência de mesma shape
            if (occupied_mask & mask) != 0:
                continue
            combined = occupied_mask | mask
            if any((combined & bm) == bm for bm in blocks):
                continue
            bad = False
            for (x,y) in cells:
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny = x+dx, y+dy
                    if 0 <= nx < board.N and 0 <= ny < board.M and grid[nx][ny] == shape:
                        bad = True; break
                if bad: break
            if bad:
                continue

            # --- forward-check de conectividade ---
            prev_mask = occupied_mask
            prev_free = free_mask

            free_mask   &= ~region_cell_mask[r]
            occupied_mask |= mask

            for (x,y) in cells:
                grid[x][y] = shape
            region_to_val[r] = v

            # se não há como manter todos pretos ligados, faz rollback imediato
            if not flood_fill_connectivity(occupied_mask | free_mask, board):
                # desfaz tudo
                occupied_mask = prev_mask
                free_mask     = prev_free
                for (x,y) in cells:
                    grid[x][y] = region_map[x][y]
                region_to_val.pop(r, None)
                continue

            # 5) desce recursivamente
            if backtrack():
                return True

            # 6) rollback normal
            occupied_mask = prev_mask
            free_mask     = prev_free
            for (x,y) in cells:
                grid[x][y] = region_map[x][y]
            region_to_val.pop(r, None)

        assigned.remove(r)
        return False

    if not backtrack():
        return region_map

    # monta e devolve o grid final
    final = [[region_map[r][c] for c in range(board.M)] for r in range(board.N)]
    for r, v in region_to_val.items():
        for (x,y) in v['cells']:
            final[x][y] = v['shape']
    return final



# ——————————————————————————————————————————————————————————————————————————————
#  MAIN: lê stdin, separa puzzles por linha em branco, resolve e imprime com tabs
# ——————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    data = sys.stdin.read()
    if not data.strip():
        sys.exit(0)

    puzzles = Board.parse_all_instances(data)
    outputs = []
    for region_map in puzzles:
        solved = solve_one_puzzle(region_map)
        lines_out = []
        for row in solved:
            lines_out.append("\t".join(str(x) for x in row))
        outputs.append("\n".join(lines_out))
    sys.stdout.write("\n\n".join(outputs))
