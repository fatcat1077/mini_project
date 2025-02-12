import random
from collections import defaultdict

# -------------------------
# 1. 定義牌的結構與相關函式
# -------------------------
SUITS = ['B', 'T', 'W']   # 筒(B)、條(T)、萬(W)
RANKS = list(range(1, 10)) # 1~9
WINDS = ['E', 'S', 'W', 'N']    # 東南西北
DRAGONS = ['Z', 'F', 'B']       # 中 發 白
FLOWERS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']  # 8 花（示意）

def create_tile_set():
    """
    建立一副台灣麻將牌（示例使用 144 張：包含 8 花）
    - 三種花色(筒/條/萬)各 1~9，四張各 = 108 張
    - 風牌(東南西北)各4張 = 16 張
    - 三元牌(中發白)各4張 = 12 張
    - 花牌 8 張
    """
    tiles = []

    # 1) 筒/條/萬，各 1~9，4 張
    for suit in SUITS:
        for rank in RANKS:
            for _ in range(4):
                tiles.append(f"{suit}{rank}")

    # 2) 風牌: 東南西北，各 4 張
    for wind in WINDS:
        for _ in range(4):
            tiles.append(wind)

    # 3) 三元牌: 中發白，各 4 張
    for dragon in DRAGONS:
        for _ in range(4):
            tiles.append(dragon)

    # 4) 花牌: 預設 8 張
    for flower in FLOWERS:
        tiles.append(flower)

    return tiles

def is_flower(tile):
    """
    判斷是否為花牌 (此處以簡單字串判斷)
    """
    return tile.startswith('F')

# --------------------------------
# 2. 定義判斷胡牌的示範函式 (16 張)
# --------------------------------
def can_win_16_hand(tiles):
    """
    判斷手牌（16 張）是否符合「5 組(刻/順) + 1 對將」。
    這是非常簡化的示例，沒有完整考慮花牌在手的情況，也沒考慮各種特殊番型。
    實際可再做更多嚴謹的檢查與演算法優化。
    """

    # 先過濾掉花牌 (花牌理論上不會放在最終判斷的 16 張中，但這裡保險處理)
    tiles_no_flower = tile
    if len(tiles_no_flower) != 17:
        return False

    # 先嘗試找「將」(pair)；抓到 pair 後，其餘 14 張要能分成 4 組刻/順。
    # 這裡提供一個簡單的「DFS 拆牌」邏輯示範。

    tile_count = defaultdict(int)
    for t in tiles_no_flower:
        tile_count[t] += 1

    unique_tiles = list(set(tiles_no_flower))

    # 檢查所有可能將牌
    for tile in unique_tiles:
        if tile_count[tile] >= 2:
            # 先取出一對當將
            tile_count[tile] -= 2
            if check_melds_14(tile_count):
                return True
            # 還原
            tile_count[tile] += 2

    return False

def check_melds_14(tile_count):
    """
    檢查在 tile_count 下的所有牌（14 張）能否拆成 4 組刻/順。
    只示範「筒/條/萬」順子判斷，以及刻子(3 張相同)，風牌/字牌無法組順子，僅能刻子。
    """
    # 若全部排都用完，即代表可以組成 4 組
    if all(c == 0 for c in tile_count.values()):
        return True

    # 找第一張數量不為 0 的牌作起始
    for tile, cnt in tile_count.items():
        if cnt > 0:
            # 1) 嘗試刻子
            if cnt >= 3:
                tile_count[tile] -= 3
                if check_melds_14(tile_count):
                    tile_count[tile] += 3
                    return True
                tile_count[tile] += 3

            # 2) 嘗試順子 (只適用筒/條/萬)
            if tile[0] in SUITS:
                suit = tile[0]      # B/T/W
                try:
                    rank = int(tile[1:])  # 1~9
                except:
                    continue

                # 檢查 rank, rank+1, rank+2
                needed_tiles = [f"{suit}{rank+i}" for i in range(3)]
                # 是否都存在
                if all(tile_count[t] >= 1 for t in needed_tiles):
                    for t in needed_tiles:
                        tile_count[t] -= 1
                    if check_melds_14(tile_count):
                        for t in needed_tiles:
                            tile_count[t] += 1
                        return True
                    # 還原
                    for t in needed_tiles:
                        tile_count[t] += 1
            break

    return False

# -------------------------
# 3. 建立玩家、遊戲流程類別
# -------------------------
class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.hand = []      # 手牌 (list)
        self.flowers = []   # 補出的花牌紀錄
        self.melds = []     # 副露 (吃/碰/槓) 的列表

    def __str__(self):
        return f"Player {self.id} | Hand: {self.hand} | Flowers: {self.flowers} | Melds: {self.melds}"

class MahjongGame:
    def __init__(self):
        self.tiles = create_tile_set()
        random.shuffle(self.tiles)

        self.players = [Player(i) for i in range(4)]
        self.current_player_idx = 0

        # 依規則：若要「莊家多一張」可在此加，但範例先不做莊家17張
        self.deal_initial_hands()

        # 用兩個指標：前端摸牌 / 後端補牌
        self.front_idx = 0
        self.back_idx = len(self.tiles) - 1

        self.discard_pile = []  # 棄牌區 (簡化)

        self.winner = None
        self.game_over = False

    def deal_initial_hands(self):
        """
        每人先抓 16 張 (若要莊家 17 張，可在此調整)
        """
        for _ in range(16):
            for p in self.players:
                tile = self.tiles[self.front_idx]
                self.front_idx += 1
                p.hand.append(tile)

        # 發完起手牌後，先執行所有玩家起手花牌補花
        for p in self.players:
            self.check_and_replace_flowers(p)

    def check_and_replace_flowers(self, player):
        """
        檢查玩家手上的花牌，若有則從牌尾補牌 (直到沒有花牌)
        """
        i = 0
        while i < len(player.hand):
            tile = player.hand[i]
            if is_flower(tile):
                player.flowers.append(tile)
                player.hand.pop(i)
                # 從牌尾補一張
                if self.front_idx <= self.back_idx:
                    replacement = self.tiles[self.back_idx]
                    self.back_idx -= 1
                    player.hand.append(replacement)
                else:
                    # 理論上不會發生：牌不夠了
                    print("Error: Not enough tiles to replace flower!")
                    self.game_over = True
                    return
            else:
                i += 1

        # 若補出的牌又是花，需繼續檢查，所以用 while 而不是 for

    def draw_tile(self, player):
        """
        玩家從牌山前端摸牌，若摸到花牌也執行補花
        """
        if self.front_idx > self.back_idx:
            # 無牌可摸 => 流局
            print("[No Tiles] 流局！")
            self.game_over = True
            return

        tile = self.tiles[self.front_idx]
        self.front_idx += 1
        player.hand.append(tile)
        # 檢查是否為花，若是就要補花
        self.check_and_replace_flowers(player)

    def discard_tile(self, player):
        """
        這裡用非常簡化的「隨機打牌」示例，真實的策略可自行設計。
        打出一張牌後，放入 discard_pile。
        """
        if not player.hand:
            return

        discard_index = random.randrange(len(player.hand))
        discard = player.hand.pop(discard_index)
        self.discard_pile.append((player.id, discard))
        # 這裡沒有立即觸發吃/碰/槓判斷，只做示例
        print(f"[Discard] Player {player.id} 打出: {discard}")

    def check_for_win(self, player):
        """
        簡單檢查玩家是否胡牌
        """
        if can_win_16_hand(player.hand):
            self.winner = player
            self.game_over = True
            print(f"玩家 {player.id} 胡了！ 手牌: {player.hand}")
        else:
            pass  # 未胡

    def step(self):
        """
        執行一個玩家的「摸牌 -> (可槓?) -> 打牌 -> (其他家可吃碰胡?)」的流程示例
        這裡先省略吃/碰/槓的多家判斷，示意用。
        """
        current_player = self.players[self.current_player_idx]

        # 1) 摸牌
        self.draw_tile(current_player)
        if self.game_over:
            return

        # 2) 檢查胡牌 (自摸)
        self.check_for_win(current_player)
        if self.game_over:
            return

        # 3) 範例中省略「是否槓牌」的判斷，可在這邊實作暗槓、加槓等
        #    若有槓，再進行補牌 (後端)

        # 4) 打牌 (目前隨機打一張)
        self.discard_tile(current_player)
        if self.game_over:
            return

        # 5) 假設其他家可以搶胡、碰、吃等，這裡先省略，只展示結構
        #    若要實作，則需要在此依序檢查「誰可以胡？誰可以槓？誰可以碰？誰可以吃？」
        #    並根據優先順序執行。

        # 最後，檢查剩餘牌數是否 <= 16 => 流局
        if (self.back_idx - self.front_idx + 1) <= 16:
            print("[流局] 牌山剩餘 16 張，結束本局。")
            self.game_over = True
            return

        # 下一位
        self.current_player_idx = (self.current_player_idx + 1) % 4

    def play(self):
        """
        不斷循環 step()，直到有人胡牌或流局。
        """
        print("---- 開始單一局模擬 (台灣16張) ----")
        round_count = 0

        while not self.game_over:
            round_count += 1
            self.step()

        print("---- 本局結束 ----")
        if self.winner:
            print(f"得勝者: Player {self.winner.id}")
        else:
            print("沒有玩家胡牌，或因流局結束。")

        # TODO: 這裡可接續做台數計算的模組 (暫不實作)
        # e.g. self.calculate_score(...)

# -------------------------
# 4. 進行單局模擬
# -------------------------
def simulate_single_hand():
    game = MahjongGame()
    game.play()

if __name__ == "__main__":
    simulate_single_hand()
