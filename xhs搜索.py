import csv
import datetime
import json
import os
import time
import urllib.parse
import ctypes
import random
import urllib.parse
from http.cookies import SimpleCookie
import bs4
import execjs
import requests
from tenacity import retry, stop_after_attempt
from loguru import logger
import os.path
import pandas
import openpyxl
from openpyxl.drawing.image import Image

class XS:
    def __init__(self, cookie: str, token: str):
        self.cookie = cookie
        self.cookies = self.cookie_str_to_dict(cookie)
        self.a1 = self.cookies['a1']
        if not self.a1:
            raise ValueError('cookie有误')
        self.b1 = 'I38rHdgsjopgIvesdVwgIC+oIELmBZ5e3VwXLgFTIxS3bqwErFeexd0ekncAzMFYnqthIhJeSBMDKutRI3KsYorWHPtGrbV0P9WfIi/eWc6eYqtyQApPI37ekmR1QL+5Ii6sdnoeSfqYHqwl2qt5B0DoIx+PGDi/sVtkIxdeTqwGtuwWIEhBIE3s3Mi3ICLdI3Oe0Vtl2ADmsLveDSJsSPw5IEvsiVtJOqw8BVwfPpdeTFWOIx4TIiu6ZPwbPut5IvlaLbgs3qtxIxes1VwHIkumIkIyejgsY/WTge7eSqte/D7sDcpipBKefm4sIx/efutZIE0ejutIbmrjPdpsI3dekut3pW7e19k4IEpzIhEe+AFDI3RPKIGnIhq9/qwMICcu+g3sdlOeVPw3Iv6e0fged0lGIi5e6pr7KVwSIkNs6B0sxcOeiVt/c9deYqwvICMyLn3sWPtSs/eeWutmICVLIvZUnqwGLPw+LjIzKS6sdoKeVVw8IEesW0AskFRYIk/sWpvskdhFIkOe6PtfIkZMOVwPtPtUI3oeTVt6IiAsVVwxIE/sfPtFtqw8sqwlIvG5Ixhg29ufKcKexVtIIhI4Ii7eSqwzrz8rLj7sWsbTIicm4PttZqwRIv5e3Ptu+DNeVz5sVDlmIhosxgeskutLLVwqIC0efVtUIiOe6UuPICKsd/3sjqwFIC6sVutbs9GtIkSKIxNsVI/ekgeedZosDuteIiF9IxkMmVtQICRFIh6s3bLXNjNs0PtqIERrBqtEJYNsjF5s0qwseuwZICYDIiZH/Vw2GqtbIvoe0zNskut4rVw2Ik4dsqtWI3WPZ0ixIiOsjgds07/eTVtzyuwQIEMbICNs3VtdIE5edFk7IvlyeMZtIxF8Ix0skPtvBqwH4uw9IC86Ik3skdoe3VwVgpGTIvud8Ptgmqt8IxqEICcGIEIRIkKs3MKs6pYENPwsICDlputocVt9Iids60EuIE4Tsa3sjqwTcutwgIiLZoIYIkTs/PwHIk3s6WQmIEqnzPtSIhi/HutvrPtBIhu7IvSEIvq8/qtDIEQPIiNe1VtEpSLVIkpZI3/sSUvsVuwYIv4VIEhUGmH1IvpAnm8hIx5eWDdejVwrIE3sTPtwIhLpIvKeYVtbIxiczut0I33eTDMGoqwx8PwsICgsYqt8IEvsjLbiIEKsxDAs6utrO/0sYVtmaWoskut8IxesiWeexutoIEVXIEFp/qtPybuaIvesjqtaL7As1FY7nVtqIx4/Ii6sDlSXI3AefqtrICqPIh/s0WGdIiEPIkDeJFmzIvOejYLXIkPAIvzMI3/edgDJJuwybVwcgboe0Y0sDqtXIvveYpOeVuwYsutNIx4yaPtTbqtp8qw3HPwiPPtENjgs1L/s6UOs3l6e6PwnIiP/IhS1I3+hI37ekgee67mUPPtOIEzOePt2sUJekutNIxge0p6sSfI4IxdejVwmwpm5IvNedqwEIx5eVVwEtPtqI3AefgIaOPttLBJeDYbWIk0sjVwmgPwYa9JsxVwbbqtNgPtZIxRyIk+horesYM6ekVwAIvAsVutKzPtrHgAskILvbUu04IQULutrePtPICTAIvzJIxbI2ZvsfjGwIvmsICMcBPw/I38SyPwgIENs3pb7gqwjIkYBZutQI3vsVY4oIhH6I3ee0utLrqwsrzgsfrOsjVwL'
        self.xs = ''
        self.xt = ''
        self.token = token
        self.lookup = [
            "Z",
            "m",
            "s",
            "e",
            "r",
            "b",
            "B",
            "o",
            "H",
            "Q",
            "t",
            "N",
            "P",
            "+",
            "w",
            "O",
            "c",
            "z",
            "a",
            "/",
            "L",
            "p",
            "n",
            "g",
            "G",
            "8",
            "y",
            "J",
            "q",
            "4",
            "2",
            "K",
            "W",
            "Y",
            "j",
            "0",
            "D",
            "S",
            "f",
            "d",
            "i",
            "k",
            "x",
            "3",
            "V",
            "T",
            "1",
            "6",
            "I",
            "l",
            "U",
            "A",
            "F",
            "M",
            "9",
            "7",
            "h",
            "E",
            "C",
            "v",
            "u",
            "R",
            "X",
            "5",
        ]
        # self.ctx1 = execjs.compile(self.sign())
        with open('xhs_0829.js', encoding='utf-8') as f:
            self.ctx1 = execjs.compile(f.read())

    @staticmethod
    def cookie_str_to_dict(cookie_str):
        cookie = SimpleCookie()
        cookie.load(cookie_str)
        return {key: morsel.value for key, morsel in cookie.items()}

    def sign(self):
        url = "http://124.223.168.69:9000/spider_api/api/js/xhs"
        headers = {
            "apiKey": self.token,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        }
        response = requests.get(url, headers=headers)
        return response.text

    def x_s(self, api, params=None, data=None):
        if params:
            api = api + f"?{urllib.parse.urlencode(params)}"
        xs = self.ctx.call("get_xs", api, data, self.a1)
        self.xs = xs['X-s']
        self.xt = str(xs['X-t'])
        return self.x_s_common()

    def x_s_71(self, api, params=None, data=None):
        if params:
            api = api + f"?{urllib.parse.urlencode(params)}"
        xs = self.ctx1.call("get_xs", api, data, self.a1)
        self.xs = xs['X-s']
        self.xt = str(xs['X-t'])
        # self.xs = xs['']
        # self.xt = str(int(time.time() * 1000))
        return self.x_s_common()

    def x_s_interface(self, api, params=None, data=None):
        if params:
            api = api + f"?{urllib.parse.urlencode(params)}"
        url = "http://124.223.168.69:9000/spider/xhs/xs"
        payload = json.dumps({
            "token": self.token,
            "cookie": self.cookie,
            "api": api,
            "data": data
        })
        headers = {
            'Content-Type': 'application/json',
        }
        xs = requests.post(url, headers=headers, data=payload).json()['data']['xs']
        self.xs = xs['X-s']
        self.xt = str(xs['X-t'])
        return self.x_s_common()

    def x_s_common(self):
        """
        takes in a URI (uniform resource identifier), an optional data dictionary, and an optional ctime parameter. It returns a dictionary containing two keys: "x-s" and "x-t".
        """
        common = {
            "s0": 5,
            "s1": "",
            "x0": "1",
            "x1": "3.7.8-2",
            "x2": "Windows",
            "x3": "xhs-pc-web",
            "x4": "4.32.0",
            "x5": self.a1,
            "x6": int(self.xt),
            "x7": self.xs,
            "x8": self.b1,
            "x9": self.mrc(self.xt + self.xs + self.b1),
            "x10": 48
        }
        encode_str = self.encode_utf8(json.dumps(common, separators=(',', ':')))
        x_s_common = self.b64_encode(encode_str)
        x_b3_traceid = self.get_b3_trace_id()
        return {
            "x-s": self.xs,
            "x-t": self.xt,
            "x-s-common": x_s_common,
            "x-b3-traceid": x_b3_traceid
        }

    @staticmethod
    def get_b3_trace_id():
        re = "abcdef0123456789"
        je = 16
        e = ""
        for t in range(16):
            e += re[random.randint(0, je - 1)]
        return e

    @staticmethod
    def mrc(e):
        ie = [
            0, 1996959894, 3993919788, 2567524794, 124634137, 1886057615, 3915621685,
            2657392035, 249268274, 2044508324, 3772115230, 2547177864, 162941995,
            2125561021, 3887607047, 2428444049, 498536548, 1789927666, 4089016648,
            2227061214, 450548861, 1843258603, 4107580753, 2211677639, 325883990,
            1684777152, 4251122042, 2321926636, 335633487, 1661365465, 4195302755,
            2366115317, 997073096, 1281953886, 3579855332, 2724688242, 1006888145,
            1258607687, 3524101629, 2768942443, 901097722, 1119000684, 3686517206,
            2898065728, 853044451, 1172266101, 3705015759, 2882616665, 651767980,
            1373503546, 3369554304, 3218104598, 565507253, 1454621731, 3485111705,
            3099436303, 671266974, 1594198024, 3322730930, 2970347812, 795835527,
            1483230225, 3244367275, 3060149565, 1994146192, 31158534, 2563907772,
            4023717930, 1907459465, 112637215, 2680153253, 3904427059, 2013776290,
            251722036, 2517215374, 3775830040, 2137656763, 141376813, 2439277719,
            3865271297, 1802195444, 476864866, 2238001368, 4066508878, 1812370925,
            453092731, 2181625025, 4111451223, 1706088902, 314042704, 2344532202,
            4240017532, 1658658271, 366619977, 2362670323, 4224994405, 1303535960,
            984961486, 2747007092, 3569037538, 1256170817, 1037604311, 2765210733,
            3554079995, 1131014506, 879679996, 2909243462, 3663771856, 1141124467,
            855842277, 2852801631, 3708648649, 1342533948, 654459306, 3188396048,
            3373015174, 1466479909, 544179635, 3110523913, 3462522015, 1591671054,
            702138776, 2966460450, 3352799412, 1504918807, 783551873, 3082640443,
            3233442989, 3988292384, 2596254646, 62317068, 1957810842, 3939845945,
            2647816111, 81470997, 1943803523, 3814918930, 2489596804, 225274430,
            2053790376, 3826175755, 2466906013, 167816743, 2097651377, 4027552580,
            2265490386, 503444072, 1762050814, 4150417245, 2154129355, 426522225,
            1852507879, 4275313526, 2312317920, 282753626, 1742555852, 4189708143,
            2394877945, 397917763, 1622183637, 3604390888, 2714866558, 953729732,
            1340076626, 3518719985, 2797360999, 1068828381, 1219638859, 3624741850,
            2936675148, 906185462, 1090812512, 3747672003, 2825379669, 829329135,
            1181335161, 3412177804, 3160834842, 628085408, 1382605366, 3423369109,
            3138078467, 570562233, 1426400815, 3317316542, 2998733608, 733239954,
            1555261956, 3268935591, 3050360625, 752459403, 1541320221, 2607071920,
            3965973030, 1969922972, 40735498, 2617837225, 3943577151, 1913087877,
            83908371, 2512341634, 3803740692, 2075208622, 213261112, 2463272603,
            3855990285, 2094854071, 198958881, 2262029012, 4057260610, 1759359992,
            534414190, 2176718541, 4139329115, 1873836001, 414664567, 2282248934,
            4279200368, 1711684554, 285281116, 2405801727, 4167216745, 1634467795,
            376229701, 2685067896, 3608007406, 1308918612, 956543938, 2808555105,
            3495958263, 1231636301, 1047427035, 2932959818, 3654703836, 1088359270,
            936918000, 2847714899, 3736837829, 1202900863, 817233897, 3183342108,
            3401237130, 1404277552, 615818150, 3134207493, 3453421203, 1423857449,
            601450431, 3009837614, 3294710456, 1567103746, 711928724, 3020668471,
            3272380065, 1510334235, 755167117,
        ]
        o = -1

        def right_without_sign(num: int, bit: int = 0) -> int:
            val = ctypes.c_uint32(num).value >> bit
            max32_int = 4294967295
            return (val + (max32_int + 1)) % (2 * (max32_int + 1)) - max32_int - 1

        for n in range(57):
            o = ie[(o & 255) ^ ord(e[n])] ^ right_without_sign(o, 8)
        return o ^ -1 ^ 3988292384

    def triplet_to_base64(self, e):
        return (
                self.lookup[63 & (e >> 18)] +
                self.lookup[63 & (e >> 12)] +
                self.lookup[(e >> 6) & 63] +
                self.lookup[e & 63]
        )

    def encode_chunk(self, e, t, r):
        m = []
        for b in range(t, r, 3):
            n = (16711680 & (e[b] << 16)) + \
                ((e[b + 1] << 8) & 65280) + (e[b + 2] & 255)
            m.append(self.triplet_to_base64(n))
        return ''.join(m)

    def b64_encode(self, e):
        p = len(e)
        w = p % 3
        u = []
        z = 16383
        h = 0
        zz = p - w
        while h < zz:
            u.append(self.encode_chunk(e, h, zz if h + z > zz else h + z))
            h += z
        if 1 == w:
            f = e[p - 1]
            u.append(self.lookup[f >> 2] + self.lookup[(f << 4) & 63] + "==")
        elif 2 == w:
            f = (e[p - 2] << 8) + e[p - 1]
            u.append(self.lookup[f >> 10] + self.lookup[63 & (f >> 4)] +
                     self.lookup[(f << 2) & 63] + "=")
        return "".join(u)

    @staticmethod
    def encode_utf8(e):
        b = []
        m = urllib.parse.quote(e, safe='~()*!.\'')
        w = 0
        while w < len(m):
            t = m[w]
            if t == "%":
                e = m[w + 1] + m[w + 2]
                s = int(e, 16)
                b.append(s)
                w += 2
            else:
                b.append(ord(t[0]))
            w += 1
        return b


class Xhs:
    def __init__(self, cookie):
        self.xhs_url = "https://edith.xiaohongshu.com"
        self.ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        self.cookie = cookie
        self.headers = {
            "authority": "edith.xiaohongshu.com",
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,zh-TW;q=0.8",
            "cache-control": "no-cache",
            "content-type": "application/json;charset=UTF-8",
            "cookie": self.cookie,
            "origin": "https://www.xiaohongshu.com",
            "pragma": "no-cache",
            "referer": "https://www.xiaohongshu.com/",
            "sec-ch-ua": "\"Google Chrome\";v=\"119\", \"Chromium\";v=\"119\", \"Not?A_Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": self.ua,
            "x-b3-traceid": "d19835f4cc6cf5a3",
        }
        self.token = '37ed6fbf7296435eaa41e2ea2df97a3a'
        self.xs = XS(cookie, self.token)

    @retry(reraise=True, stop=stop_after_attempt(3))
    def req(self, api, params=None, data=None):
        if not data:
            return requests.get(self.xhs_url + api if 'http' not in api else api, headers=self.headers, params=params, timeout=3)
        data = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return requests.post(self.xhs_url + api if 'http' not in api else api, headers=self.headers, params=params, data=data, timeout=3)

    def get_xs(self, api, params=None, data=None):
        if params:
            api = api + f"?{urllib.parse.urlencode(params)}"
        xs = self.xs.x_s_71(api, data=data)
        self.headers["x-s"] = xs['x-s']
        self.headers["x-t"] = xs['x-t']
        self.headers['x-s-common'] = xs['x-s-common']
        self.headers["x-b3-traceid"] = xs["x-b3-traceid"]
        # self.headers['x-s-common'] = "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"
        # self.headers["x-b3-traceid"] = "01fef972a7209078"

    def search(self, keyword, page, sort=1, note_type=0):
        """
        小红书搜索
        :param keyword: 关键字
        :param page: 页码
        :param sort: 1-默认 2-最热 3-最新
        :param note_type: 0-全部 1-视频 2-图文
        :return:
        """
        api = '/api/sns/web/v1/search/notes'
        data = {
            "keyword": keyword,
            "page": page,
            "page_size": 20,
            "search_id": "2c6oczuciv55hg9uuy6li@2c6od1b9741n574k7n79q",
            "sort": "general" if sort == 1 else 'popularity_descending' if sort == 2 else 'time_descending',
            "note_type": note_type,
            "image_scenes": "FD_PRV_WEBP,FD_WM_WEBP"
        }
        self.get_xs(api, data=data)
        response = self.req(api, data=data)
        return response.json()

    @staticmethod
    def search_(note, data=None):
        if not data:
            data = {}
        if 'note' not in note['model_type']:
            return
        note_id = note['id']
        data['笔记id'] = note_id
        data['token'] = note['xsec_token']
        note = note['note_card']
        data['标题'] = note.get('display_title')
        data['详情链接'] = f"https://www.xiaohongshu.com/explore/{data['笔记id']}?xsec_token={data['token']}"
        data_dir = f"./data/{note_id}"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data

    def feed(self, note_id,token):
        api = '/api/sns/web/v1/feed'
        data = {
            "extra": {
              "need_body_topic": "1"
            },
            "image_formats": [
              "jpg",
              "webp",
              "avif"
            ],
            "source_note_id": note_id,
            "xsec_source": "pc_search",
            "xsec_token": token,
        }
        self.get_xs(api, data=data)
        response = self.req(api, data=data)
        return response.json()

    def feed_(self, feed_data=None, data=None):
        if not feed_data and not data:
            raise ValueError('笔记详情参数有误')
        if not feed_data:
            feed_data = self.feed(data['笔记id'],data['token'])['data']['items'][0]
        if not data:
            data = {}
        data['笔记id'] = feed_data['id']
        feed_data = feed_data['note_card']
        data['标题'] = feed_data.get('title')
        data['描述'] = feed_data.get('desc')
        data['笔记类型'] = feed_data['type']
        data['发布时间'] = datetime.datetime.fromtimestamp(feed_data['last_update_time'] / 1000)
        data['点赞数'] = feed_data['interact_info']['liked_count']
        data['收藏数'] = feed_data['interact_info']['collected_count']
        data['转发数'] = feed_data['interact_info']['share_count']
        data['评论数'] = feed_data['interact_info']['comment_count']
        data['作者id'] = feed_data['user']['user_id']
        data['作者昵称'] = feed_data['user']['nickname']
        imageList = feed_data['image_list']
        index = 1
        for ima in imageList:
            if ima.get('url_pre') is None:
                continue
            self.downlaod(ima.get('url_pre'), data['笔记id'], index)
            index += 1
        return data

    def comments(self, note_id, cursor=''):
        api = '/api/sns/web/v2/comment/page'
        params = {
            "note_id": note_id,
            "cursor": cursor,
            "top_comment_id": "",
            "image_formats": "jpg,webp,avif"
        }
        self.get_xs(api, params=params)
        response = self.req(api, params=params)
        return response.json()

    @staticmethod
    def comments_(comment, data=None):
        if not data:
            data = {}
        data['评论id'] = "'"+comment['id']
        data['评论内容'] = comment['content']
        data['评论时间'] = datetime.datetime.fromtimestamp(comment['create_time'] / 1000)
        data['用户IP'] = comment.get('ip_location')
        data['点赞数'] = comment['like_count']
        data['用户名'] = comment['user_info']['nickname']
        data['用户id'] = comment['user_info']['user_id']
        return data

    def sub_comments(self, note_id, root_comment_id, cursor):
        api = '/api/sns/web/v2/comment/sub/page'
        params = {
            "note_id": note_id,
            "root_comment_id": root_comment_id,
            "num": "10",
            "cursor": cursor,
            "image_formats": ""
        }
        self.get_xs(api, params=params)
        response = self.req(api, params=params)
        return response.json()

    def comments_run(self, note_id, cursor='', csv_file=''):
        total = []
        while True:
            od = []
            logger.info(f"{note_id, cursor}")
            comments = self.comments(note_id, cursor)
            if comments.get('data') is None or comments.get('data').get('comments') is None:
                continue
            for c in comments['data']['comments']:
                data = self.comments_(c, {"note_id": note_id})
                print(data)
                total.append(data)
                od.append(data)
                if int(c['sub_comment_count']) != 0:
                    sub_cursor = ''
                    while True:
                        logger.info(f"子评论 {sub_cursor} ")
                        sub_comments = self.sub_comments(note_id, c['id'], sub_cursor)
                        for sub in sub_comments['data']['comments']:
                            data = self.comments_(sub, {"note_id": note_id})
                            print(data)
                            total.append(data)
                            od.append(data)
                        if not sub_comments['data']['has_more']:
                            break
                        sub_cursor = sub_comments['data']['cursor']
            if csv_file:
                self.write(od, csv_file)
            if not comments['data']['has_more']:
                break
            cursor = comments['data']['cursor']
        return total

    def user_posted(self, user_id, cursor):
        api = '/api/sns/web/v1/user_posted'
        params = {
            "num": "30",
            "cursor": cursor,
            "user_id": user_id,
            # "image_scenes": "FD_PRV_WEBP,FD_WM_WEBP",
            "image_formats": "jpg,webp,avif"
        }
        self.get_xs(api, params=params)
        response = self.req(api, params=params)
        return response.json()

    def user_top_posted_ids(self, user_id):
        url = f"https://www.xiaohongshu.com/user/profile/{user_id}"
        response = self.req(url)
        html = bs4.BeautifulSoup(response.text, 'lxml')
        note_ids = []
        for t in html.find_all(class_='top-tag-area'):
            note_id = t.parent.get('href').split('/')[-1]
            note_ids.append(note_id)
        return note_ids

    def user_posted_run(self, uid, csv_file='',cursor=''):
        tt = []
        try:
            while True:
                od = []
                logger.info(f"[用户笔记] uid: {uid} cursor: {cursor}")
                posted = self.user_posted(uid, cursor)
                # if posted.get('data') is None:
                #     break
                for post in posted['data']['notes']:
                    data = self.feed_(data={"笔记id": post['note_id'],"token": post['xsec_token']})
                    print(data)
                    od.append(data)
                    tt.append(data)
                    # time.sleep(2)
                if csv_file:
                    self.write(od, csv_file)
                if not posted['data']['has_more']:
                    break
                cursor = posted['data']['cursor']
        except Exception as e:
            return False, [uid, cursor], e
        return True, tt, None

    def search_run(self, keyword, begin=1, end=5, sort=1, note_type=0, csv_file=''):
        for page in range(begin, end + 1):
            od = []
            logger.info(f"{keyword, page}")
            search = self.search(keyword, page, sort, note_type)
            for note in search['data']['items']:
                data = self.search_(note, data={'搜索关键字': keyword})
                if not data:
                    continue
                self.feed_(data=data)
                print(data)
                od.append(data)
                time.sleep(3)
            if csv_file:
                self.write(od, csv_file)
            if not search['data']['has_more']:
                break
            time.sleep(1)

    def person_info(self,data):
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "cookie": cookie_str,
            "priority": "u=0, i",
            "referer": "https://www.xiaohongshu.com/explore/6728cbac000000001b0110ed?xsec_token=ABL5vYak3WR9_A8FMdFIAk43Kw_UO5W7AZzR94Qax-dhQ=&xsec_source=pc_feed",
            "sec-ch-ua": "\"Chromium\";v=\"130\", \"Google Chrome\";v=\"130\", \"Not?A_Brand\";v=\"99\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
        url = f"https://www.xiaohongshu.com/user/profile/{data['作者id']}"
        params = {
            "xsec_token": "",
            "xsec_source": "pc_note"
        }
        response = requests.get(url, headers=headers, params=params)
        datas = self.find_str(response.text, "window.__INITIAL_STATE__=", '</script>')
        personInfo = json.loads(datas.split('"userPageData":')[-1].split(',"activeTab"')[0])
        for i in personInfo.get('interactions'):
            data[i['name']] = i['count']
        data['标签'] = '; '.join([ta.get('name') if ta.get('name') else '' for ta in personInfo.get('tags')])
        data['小红书号'] = personInfo.get('basicInfo').get('redId')

    @staticmethod
    def find_str(input_str, left_str, right_str):
        left_index = input_str.find(left_str)
        if left_index == -1:
            return None
        right_index = input_str.find(right_str, left_index + len(left_str))
        if right_index == -1:
            return None
        return input_str[left_index + len(left_str):right_index]

    @staticmethod
    def write(data, path):
        if data:
            with open(path + '.csv', 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(data)

    @staticmethod
    def r_write(data, path):
        if data:
            with open(path + '.csv', 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(data)

    @staticmethod
    def read_csv(path):
        data: [dict] = []
        with open(path + '.csv', 'r', newline='', encoding='utf-8-sig') as f:
            for d in csv.DictReader(f):
                data.append(d)
        return data

    @staticmethod
    def write_s(out_data, excel_file):
        if not out_data:
            return
        df = pandas.DataFrame(out_data)
        df.to_excel(excel_file, index=False)

        wb = openpyxl.load_workbook(excel_file)
        ws = wb.active

        for i, v in enumerate(out_data):
            try:
                name = v['笔记id'].split('/')[-1].split('.jpg')[0]
                img = Image(f"./data/{name}/{0}.jpg")
            except:
                continue
            img.width, img.height = (140, 120)
            ws.column_dimensions['A'].width = 20
            ws.row_dimensions[i + 2].height = 70
            ws.add_image(img, anchor='A' + str(i + 2))

        wb.save(excel_file)
        wb.close()

    @staticmethod
    def downlaod(imageurl,id,name):
        try:
            if not os.path.exists(f"./data/{id}"):
                os.makedirs(f"./data/{id}")
            response = requests.get(imageurl, timeout=5)
            image_path = f'{name}.jpeg'
            with open(f"./data/{id}/{image_path}", 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    cookie_str = 'abRequestId=62e93589-1432-52ef-87c3-925e59b523fc; xsecappid=xhs-pc-web; a1=193a0032d2do9z69de9irp2hq4hpsery0sqivcy2g50000310429; webId=e3b8bcb78161edb27a86d1de9d225ff3; gid=yjq088qqJJ8Dyjq088qJfMSvJfljuKjfdj3FAJkUJxA4x328UMdFvh888qy84Jj88i2f4Yj2; web_session=040069b3f1bd0fd4f202d0b066354b6454432c; webBuild=4.46.0; websectiga=8886be45f388a1ee7bf611a69f3e174cae48f1ea02c0f8ec3256031b8be9c7ee; sec_poison_id=f0a81152-1f8c-4e65-9532-0a740d4a769a; unread={%22ub%22:%2267381f88000000001901430e%22%2C%22ue%22:%226747eb2500000000070396c8%22%2C%22uc%22:25}; acw_tc=0a0bb14717335777052315520e65d179912a69fd93d63e16c95f14b53d665f'
    keyword = '大学生就业' #这里是关键词
    xhs = Xhs(cookie_str)
    xhs.search_run(keyword, 1, 11, 1, 0, keyword)

