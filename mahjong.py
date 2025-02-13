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
    - 花牌 8 張 = 8 張
    總計 144 張
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
    """判斷是否為花牌"""
    return tile.startswith('F')


# --------------------------------
# 2. 定義判斷胡牌 (17 張) 的函式
# --------------------------------

def can_win_17_hand(tiles):
    """
    判斷手牌（17 張）是否符合「5 組(刻/順) + 1 對將」的基本結構。
    - 先過濾花牌（理論上花應該即時補掉，不會留在手上）
    - 若過濾後實際牌張數 != 17，就直接 False
    - 拆法：先找一對將，剩 15 張檢查能否拆成 5 組刻/順
    """
    # 過濾花牌
    tiles_no_flower = [t for t in tiles if not is_flower(t)]
    if len(tiles_no_flower) != 17:
        return False

    tile_count = defaultdict(int)
    for t in tiles_no_flower:
        tile_count[t] += 1

    unique_tiles = list(set(tiles_no_flower))

    # 嘗試所有可能的將
    for tile in unique_tiles:
        if tile_count[tile] >= 2:
            # 先取出一對當將
            tile_count[tile] -= 2
            if check_melds_15(tile_count):
                return True
            # 還原
            tile_count[tile] += 2

    return False

def check_melds_15(tile_count):
    """
    檢查在 tile_count 下的所有牌（15 張）能否拆成 5 組刻/順 (每組 3 張)。
    邏輯：遞迴 / DFS
     - 若全部用完 => True
     - 嘗試刻子或順子 => 成功即 True，失敗回溯繼續
    """
    # 若所有牌都已用完，表示成功拆出所有組合
    if all(c == 0 for c in tile_count.values()):
        return True

    # 找第一張數量不為 0 的牌作起始
    for tile, cnt in tile_count.items():
        if cnt > 0:
            # 1) 嘗試刻子
            if cnt >= 3:
                tile_count[tile] -= 3
                if check_melds_15(tile_count):
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

                needed_tiles = [f"{suit}{rank+i}" for i in range(3)]
                # 是否都存在
                if all(tile_count[t] >= 1 for t in needed_tiles):
                    for t in needed_tiles:
                        tile_count[t] -= 1
                    if check_melds_15(tile_count):
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
        self.flowers = []   # 花牌區 (補出的花)
        self.melds = []     # 副露 (吃/碰/槓)

    def __str__(self):
        return f"Player {self.id} | Hand: {self.hand} | Flowers: {self.flowers} | Melds: {self.melds}"

class MahjongGame:
    def __init__(self):
        self.tiles = create_tile_set()
        random.shuffle(self.tiles)

        self.players = [Player(i) for i in range(4)]
        self.current_player_idx = 0

        # 發牌：每人 16 張
        self.deal_initial_hands()

        # 前端摸牌 / 後端補牌 指標
        self.front_idx = 16 * 4  # 發完 16*4=64 張後，接下來摸牌的位置
        self.back_idx = len(self.tiles) - 1

        self.discard_pile = []  # 棄牌區 (簡化)

        self.winner = None
        self.game_over = False

    def deal_initial_hands(self):
        """每位玩家先抓 16 張，並進行起手補花"""
        for _ in range(16):
            for p in self.players:
                p.hand.append(self.tiles[self.front_idx])
                self.front_idx += 1

        # 檢查起手花牌並即時補花
        for p in self.players:
            self.check_and_replace_flowers(p)

    def check_and_replace_flowers(self, player):
        """
        檢查玩家手上的花牌，若有則從牌尾補牌 (直到沒有花牌為止)。
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
                    print("Error: Not enough tiles to replace flower!")
                    self.game_over = True
                    return
            else:
                i += 1
        # 若新補的牌又是花牌 => 繼續檢查，因此使用 while 而不是 for

    def draw_tile(self, player):
        """
        玩家從牌山前端摸牌，若摸到花牌則從後端補。
        此時玩家會有 17 張牌 (若原本手牌 16)。
        """
        if self.front_idx > self.back_idx:
            # 牌已摸完 => 流局
            print("[No Tiles] 流局！")
            self.game_over = True
            return

        tile = self.tiles[self.front_idx]
        self.front_idx += 1
        player.hand.append(tile)

        # 檢查補花
        self.check_and_replace_flowers(player)

    def discard_tile(self, player):
        """
        這裡用非常簡化的隨機打牌示例，
        實際可改為 AI 或規則式打牌。
        """
        if not player.hand:
            return

        discard_index = random.randrange(len(player.hand))
        discard = player.hand.pop(discard_index)
        self.discard_pile.append((player.id, discard))
        print(f"[Discard] Player {player.id} 打出: {discard}")

    def check_for_win(self, player):
        """
        簡單檢查玩家是否胡牌 (17 張)，
        不含特殊牌型。
        """
        if can_win_17_hand(player.hand):
            self.winner = player
            self.game_over = True
            print(f"玩家 {player.id} 胡牌！ 手牌: {player.hand}")
        else:
            pass  # 未胡

    def step(self):
        """
        執行單位回合流程：
          1) 摸牌 (16->17)
          2) 自摸判斷
          3) (可擴充槓牌判斷)
          4) 打牌 (17->16)
          5) 他家可吃碰槓胡 => 範例略 (僅預留結構)
          6) 剩餘牌 <= 16 => 流局
        """
        current_player = self.players[self.current_player_idx]

        # 1) 摸牌
        self.draw_tile(current_player)
        if self.game_over:
            return

        # 2) 自摸判斷
        self.check_for_win(current_player)
        if self.game_over:
            return

        # 3) 暗槓 / 加槓 等 => 可在此實作
        #    若有槓 => 需要從牌尾補 => 再次檢查胡牌
        #    這裡暫時省略

        # 4) 打牌 (回到 16 張)
        self.discard_tile(current_player)
        if self.game_over:
            return

        # 5) 其他玩家是否能吃碰槓胡 => 這裡省略示例
        #    若有人搶胡 => 中斷

        # 6) 流局判斷：牌山剩餘 <= 16 張
        if (self.back_idx - self.front_idx + 1) <= 16:
            print("[流局] 牌山剩餘 16 張，結束本局。")
            self.game_over = True
            return

        # 下一位玩家
        self.current_player_idx = (self.current_player_idx + 1) % 4

    def play(self):
        """
        不斷輪流 step()，直到有人胡牌或流局。
        """
        print("---- 開始單局模擬 (台灣 16 張) ----")
        round_count = 0

        while not self.game_over:
            round_count += 1
            self.step()

        print("---- 本局結束 ----")
        if self.winner:
            print(f"得勝者: Player {self.winner.id}")
        else:
            print("沒有玩家胡牌，或因流局結束。")
        # TODO: 台數計算部分可在這裡擴充

# -------------------------
# 4. 進行單局模擬
# -------------------------
def simulate_single_hand():
    game = MahjongGame()
    game.play()

if __name__ == "__main__":
    simulate_single_hand()
