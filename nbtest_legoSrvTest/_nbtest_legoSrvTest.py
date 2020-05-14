# coding=utf-8
from __future__ import unicode_literals, print_function


import json, types, requests, sys, codecs, traceback, threading, random, functools, inspect, re
import urlparse, urllib2
import datetime, time   # time 模块不能删，用作eval('time.sleep')
from types import NoneType
from collections import OrderedDict
import logging


from DictObject import DictObject

from nbtest import Utils
from nbtest.jsonpath import jpFinds
from nbtest.Utils import UndefCls, json_o2t, TYPE, str_fmt, str_fmtB
from nbtest.assertpyx import AX
from flask import Flask, request, jsonify, Response, helpers
import flask


UserLibs = DictObject()


class Libs(object):
    """ 【注：Libs必须是类形式，而不能是模块形式，因为模块形式在外界需要动态更新内部属性时很难生效】
    PC->ENV->INI 常量划分为三个层次，可变程度依次递增
    """
    class _ConstPC(object):
        """ ProdConst项目产品级常量(在产品中固定不变的) """
        Url_Ex = "http://127.0.0.1:9901"
        ENV_DEFAULT = 'default'

    PC = _ConstPC
    @staticmethod
    def updPC(pcCls, m=None):
        AX(pcCls, '').doCalled(Utils.isSubCls, Libs._ConstPC).is_true()
        Libs.PC = pcCls
        Libs.logger.info('updPC: Libs.PC = {}'.format(Libs.PC))

    class _ConstENV(object):
        """ ProdConst环境级常量(在产品中不同集群or不同服务器or不同app下会变的) """
        _CfgEnvs = DictObject({})

        @classmethod
        def loadCfgEnv(cls, envName, m=None):
            AX(envName, 'need envName in cfgEnvs.keys()').isIn(cls._CfgEnvs.keys())
            cfgEnv = cls._CfgEnvs[envName]
            Libs.updENV(cfgEnv, m=m)
            return TYPE and cls() and cfgEnv

        def __init__(self, __Name__=None, **kwargs):
            if __Name__ is None:
                return
            AX(__Name__, 'need __Name__ not exists in cfgEnvs.keys()').isNotIn(self.__class__._CfgEnvs.keys())
            self.__Name__ = __Name__
            self.__class__._CfgEnvs[__Name__] = self

            for k, v in kwargs.items():
                if hasattr(self, k):
                    default_v = getattr(self, k)
                    if isinstance(default_v, Libs._ConstENV):
                        Libs.logDbg('jump ')
                        continue  # TODO: 当默认值为envSon形式时 可以允许在覆盖时自定义覆盖
                    AX(v, '(new value need same type of default {}.{})'.format(self, k)).is_instance_of(type(default_v))

                setattr(self, k, v)

    ENV = _ConstENV()
    @staticmethod
    def updENV(envInst, m=None):
        AX(envInst, '').is_instance_of(Libs._ConstENV)
        Libs.ENV = envInst
        Libs.logger.info('updENV: Libs.ENV = {}'.format(Libs.ENV))

    class _ConstINI(object):  # 用户启动时配置(每次)
        """ 用户启动时配置(类似于sys.argv，用户每次启动都可能变化的) """
        EnvName = 'default'
        TryTimes = 1
        TimeoutShort = 60
        Timeout = 60 * 2
        TimeoutMedium = 60 * 5
        TimeoutLong = 60 * 10

    INI = _ConstINI
    @staticmethod
    def updINI(iniCls, m=None):
        AX(iniCls, '').doCalled(Utils.isSubCls, Libs._ConstINI).is_true()
        Libs.INI = iniCls
        Libs.logger.info('updINI: Libs.INI = {}'.format(Libs.INI))

    Code_Suc = 200  # testFlow成功码
    Code_Err = -200  # testFlow失败码
    Undefined = str(Utils.UndefCls('Undefined'))  # 模拟js的undefined且转为str

    class MyLogger(object):
        TRACE = logging.DEBUG - 1
        """ 日志类: stdout+logFile 双写 """

        def __init__(self, name='nbtest_legoSrvTest', console=True, file='',
                     consoleLv=logging.INFO, fileLv=logging.DEBUG,
                     fmt='%(asctime)s %(levelname)s [%(name)s] %(message)s'
                     ):
            # 创建一个logger
            self.logger = logging.getLogger(name)
            self.logger.setLevel(min(consoleLv, fileLv))
            formatter = logging.Formatter(fmt)

            if not (console or file):
                assert False, 'MyLogger: need (console or file)'

            # 创建一个handler，用于写入日志文件
            if file:
                self.logger.FH = logging.FileHandler(file)
                self.logger.FH.setLevel(fileLv)
                self.logger.FH.setFormatter(formatter)
                self.logger.addHandler(self.logger.FH)
            if console:
                self.logger.CH = logging.StreamHandler()
                self.logger.CH.setLevel(consoleLv)
                self.logger.CH.setFormatter(formatter)
                self.logger.addHandler(self.logger.CH)

    logger = MyLogger().logger
    logDbg = logger.debug

    @staticmethod
    def name(o, default):
        if hasattr(o, '__name__'):
            return o.__name__
        elif hasattr(o, '__Name__'):
            return o.__Name__
        elif hasattr(o, '__NAME__'):
            return o.__NAME__
        elif hasattr(o, 'name'):
            return o.name
        elif hasattr(o, 'Name'):
            return o.Name
        elif hasattr(o, 'NAME'):
            return o.NAME
        else:
            return default

    @staticmethod
    def dictToJson(dict_, dict_Name, **kwargs):
        assert isinstance(dict_Name, basestring), 'need isinstance(dict_Name, basestring), but is {!r}'.format(dict_Name)
        retJson = DictObject({})
        if isinstance(dict_, Utils.JsonLike):
            dict_ = (TYPE and Utils.JsonLike and dict_).toJson()
        for k, v in dict_.items():
            if kwargs.get('__testFlowObj__') and k == '__testFlowObj__':
                continue
            Utils.isJsonItem(v, raiseName='{}.{}'.format(dict_Name,k))
            retJson[k] = v if not isinstance(v, Utils.JsonLike) else v.toJson(k + '.')
        return retJson

    class HttpjsonResp(object):
        def __init__(self, RES=None, reqUrl=None, reqJson=None, errObj=None):
            self.reqUrl = reqUrl
            self.reqJson = reqJson
            # assert isinstance(RES, dict), 'assert isinstance(RES, dict), but isa {}, RES={}'.format(type(RES), RES)
            self.RES = RES
            self.errObj = errObj

    @staticmethod
    def jpFinds(*args, **kwargs):
        return jpFinds(*args, **kwargs)

    @staticmethod
    def httpjson_post(reqUrl, reqJson={}, reqMethod='post', reqExtKw={},
                      tryTimes=None, timeout=None,
                      __ThreadRets__=None, __ThreadName__=None, **kwargs):
        _resp = None
        errObj = None
        timeout = timeout or Libs.INI.Timeout
        RES = None

        testFlowLike = hasattr(reqUrl, '__RouteTestFlow__')  # Utils.isinstanceT(reqUrl, TestFlow.TestFlowLikeFn)
        times = 1 if testFlowLike else (tryTimes or Libs.PC.TryTimes)
        for i_times in range(times):
            RES = None
            errObj = None
            try:
                Libs.logDbg('    --><{}>: {} timeout={} tryTimes=({}/{})'.format(
                    threading.current_thread().name, reqUrl, timeout, i_times, times
                ))
                if testFlowLike:  # reqUrl是一个testFlowLike函数时

                    _resp = reqUrl(IN=FlaskExt.TempVar(**reqJson))  # @FlaskExt.routeTestFlow()
                    _resp.text = _resp.data
                    # assert _resp.status_code == 200, '<testFlowLike> status_code == 200, but status_code={}'.format(_resp.status_code)
                    RES = _resp.json
                else:
                    if isinstance(reqJson, basestring):
                        reqJson = Utils.encode_to(reqJson, 'utf-8')  # 默认编码为utf-8
                        _kwargs = dict(method=reqMethod, url=reqUrl, data=reqJson, timeout=timeout, **reqExtKw)
                        _resp = requests.request(**_kwargs)
                    else:
                        _resp = requests.request(method=reqMethod, url=reqUrl, json=reqJson, timeout=timeout,
                                                 **reqExtKw)
                    # assert _resp.status_code == 200, 'status_code == 200, but status_code={}'.format(_resp.status_code)
                    # Libs.logDbg('_resp.headers={}'.format(_resp.headers))
                    RES = _resp.json() if _resp.headers['Content-Type'].find('application/json') != -1 \
                        else dict(__TEXT__=_resp.text)
                break
            except Exception as errObj:
                if '{!r}'.format(errObj).find('timeout') != -1 or '{!r}'.format(errObj).find('status_code') != -1:
                    continue
                if _resp == None:
                    _resp = DictObject(status_code=None, text=None, errMsg=str_fmt(errObj))
                RES = {
                    '__error_at_respObj__': dict(status_code=_resp.status_code, text=_resp.text, reqUrl=str(reqUrl), errMsg=str_fmt(errObj))
                }
                break
        if RES is None:
            RES = {
                '__error_at_respObj__': dict(status_code=_resp.status_code, text=_resp.text, reqUrl=str(reqUrl))
            }
            Libs.logDbg(RES)
        ret = Libs.HttpjsonResp(
            reqUrl=reqUrl, reqJson=reqJson, RES=RES, errObj=Exception(Utils.err_detail(errObj)) if errObj else None
        )
        if __ThreadRets__:
            __ThreadRets__.addRet(__ThreadName__ or '{}--{}'.format(reqUrl, id(_resp)), ret)
        return ret

    @staticmethod
    def failMsg(failKey='', failMsg='', failStackAppendLines=[]):
        if (not failKey) and (not failMsg):
            return ''
        if isinstance(failMsg, Exception):
            failMsgNewLines = Utils.err_detail(failMsg).strip('\n').split('\n')
            failMsgNewLine0 = failMsgNewLines[0]
            if len(failStackAppendLines) != 0:
                failMsgNewLines = [failMsgNewLine0] + failStackAppendLines + failMsgNewLines[1:]
            return '{}: failMsg=\n{}'.format(failKey, '\n'.join(failMsgNewLines))
        else:
            return '{}: failMsg=\n{}'.format(failKey, failMsg)

    @staticmethod
    def to_RES(RES={}):
        if isinstance(RES, Utils.JsonLike):
            RES = RES.toJson()
        assert isinstance(RES, (dict, Utils.JsonLike)), \
            'assert isinstance(RES, (dict, Utils.DictLike)), but isa {}, RES={}'.format(type(RES), RES)
        if RES.has_key('failMsg') and not RES.get('failMsg'):
            RES.pop('failMsg')
        return RES

    class ThreadRets(object):
        def __init__(self):
            self._rets = {}

        def addRet(self, key, val):
            assert isinstance(key, basestring), \
                'isinstance(key, basestring), but type(key)={}'.format(type(key))
            self._rets[key] = val

        def lenRets(self):
            return len(self._rets)

        def rets(self):
            return self._rets

    @staticmethod
    def threadsRun(runOneFn, runOneKwargsDict, timeoutSum=None, name='', ifRaiseTimeout=True, __ThreadGap__=0.5):
        """ exam:
        threadsRun(httpjson_post, runOneKwargsDict={
            "item_id_1": {reqUrl=Url_Ex+"/testAnItem_FastAudit",reqJson=dict(item_id="item_id_1",item_type="kVideoSet4Many")},
            "item_id_2": {reqUrl=Url_Ex+"/testAnItem_FastAudit",reqJson=dict(item_id="item_id_1",item_type="kVideoSet4Many")}
        })

        """
        timeoutSum = timeoutSum or Libs.INI.TimeoutLong
        __ThreadRets__ = Libs.ThreadRets()
        threads = {}
        _names = runOneKwargsDict.keys()
        _num = len(_names)
        for i in range(_num):
            _name = _names[i]
            runOneKwargs = runOneKwargsDict[_name]
            runOneKwargs.update(
                dict(__ThreadRets__=__ThreadRets__, __ThreadName__=_name, __ThreadGap__=__ThreadGap__ * i))
            threads[_name] = threading.Thread(
                name=_name,
                target=runOneFn,
                kwargs=runOneKwargs
            )
        for _name, t in threads.items():
            t.setDaemon(True)
            t.start()

        _timeSum = timeoutSum + (__ThreadGap__ * _num)
        _timeLogGap = int(_timeSum / 100.0) or 1
        for sec in range(1, int(1 + timeoutSum + (__ThreadGap__ * _num))):
            if __ThreadRets__.lenRets() == len(threads) or (not len([t for note, t in threads.items() if t.isAlive()])):
                break
            time.sleep(1)
            if sec % 10 == 0:
                Libs.logDbg('--><{}>: threadsRun({}): ({}/<{}+{}*{}>)s'.format(threading.current_thread().name, name, sec,
                                                                          timeoutSum, __ThreadGap__, _num))
        else:
            if ifRaiseTimeout:
                raise Exception('threadsRun timeout')
        return __ThreadRets__.rets()


class TestDriver(object):
    @staticmethod
    def isaDvMethod(o):
        return hasattr(o, 'im_self') and isinstance(o.im_self, TestDriver)
    class DvDefaultRET(Utils.JsonLike):
        @property
        def RET(self): return self._RET
        def __init__(self, _RET):
            self._RET = _RET

class TestBizKw(object):
    def __init__(self, *args, **kwargs):
        self.__Inited__ = False

    def __Init__(self, *args, **kwargs):
        Libs.logger.info('{}: when not self.__Inited__ auto run self.__Init__()'.format(self))
        self.__Inited__ = True

    def __call__(self):
        return self
    @staticmethod
    def isaBizKwMethod(o):
        return hasattr(o, 'im_self') and isinstance(o.im_self, TestBizKw)

class TestStructAbstr(object):
    """ __init__: (f._initSons, f._pathSet)
        -> f.addSon(s)
            -> s._fatherSet(f) -> s.father._delSon(s) -> s._pathSet
    """

    _Father_Undef = UndefCls('TestStructAbstr._father')
    _Path_Sep = '/'
    def __init__(self, path='', **kwargs):
        self._pathSet(path)
        self._father = TestStructAbstr._Father_Undef
        self._initSons()

    @property
    def name(self):
        AX(self._name, '{}.name'.format(self)).is_instance_of(basestring)
        return TYPE and str and self._name

    @property
    def path(self):
        return TYPE and str and self._path

    def _pathSet(self, value):  # 只在 _fatherSet时才应该自动调用路径设置
        AX(value, '{}.path'.format(self)).is_instance_of(basestring)
        self._path = value
        self._name = self._path.split(TestStructAbstr._Path_Sep)[-1]

    def pathJoin(self, paths):
        return TestStructAbstr._Path_Sep.join(paths)

    @property
    def father(self):
        return TYPE and TestStructAbstr() and self._father

    def _fatherSet(self, v, newName=None):
        AX(v, '{}.father'.format(self)).is_instance_of(TestStructAbstr)
        if self._father != TestStructAbstr._Father_Undef:
            self._father._delSon(v.name)
        self._father = v
        self._pathSet( self.pathJoin((self._father.path, newName or self.name)) )   # addSon时才知道路径变化

    @property
    def sons(self):
        return TYPE and DictObject() and self._sons

    def _initSons(self):
        AX(self, '{}._initSons() need not self._sons'.format(self)).doCalled(hasattr, '_sons').is_false()
        self._sons = DictObject()

    def _delSon(self, k):
        AX(k, '{}._delSon({}) need exists'.format(self, k)).isIn(self._sons.keys())
        self._sons.pop(k)

    def addSon(self, k, v):
        AX(self._Path_Sep, '{}.addSon({}, v) cannot has({})'.format(self, k, self._Path_Sep)).isNotIn(k)
        AX(k, '{}.addSon({}, v) cannot duplic'.format(self, k)).isNotIn(self._sons.keys())
        AX(v, '{}.addSon({}, v)'.format(self, k)).is_instance_of(TestStructAbstr)
        self._sons[k] = v
        v._fatherSet(self, newName=k)


class TestFlow(TestStructAbstr):
    """ base_class for TestCase TestSuite TestModule """
    def __init__(self, name, timeout=None, **kwargs):
        TestStructAbstr.__init__(self, path=name)
        self.timeout = timeout
    def run(self):
        pass
    class TestFlowLikeFn(object):
        @classmethod
        def __IsinstanceT__(cls, o):
            return hasattr(o, '__asTestFlowFn__') and Utils.isSubCls(o.__asTestFlowFn__, TestFlow)
    @staticmethod
    def IN_kwargs(IN):
        IN_kwargs = dict(**IN)
        if IN_kwargs.get('IN')==None:
            IN_kwargs['IN'] = IN
        return DictObject(IN_kwargs)
    @staticmethod
    def GetJson(run_res):
        if isinstance(run_res, flask.wrappers.Response):
            return run_res.json
        else:
            return run_res

class TestFlowFail(TestStructAbstr):
    def __init__(self, name, failMsg):
        TestStructAbstr.__init__(self, path=name)
        self.failMsg = failMsg
    def toJson(self, **kwargs):
        return Libs.dictToJson(dict(
            path=self.path+'.toJson()', failMsg=self.failMsg
        ), self.path+'.toJson()', **kwargs)


def asTestFlowFn(fn):
    """ ①自动校验必选参:  T或者IN必须有一项 **kwargs必须有

    """
    name = fn.__name__
    fn_argSpec = inspect.getargspec(fn)
    AX(fn_argSpec, name + ':fn_argSpec').doAttrs(['varargs', 'keywords']).\
        isItemsEq(dict(varargs=None, keywords="kwargs"))
    fn_args = fn_argSpec.args[1:] if (fn_argSpec.args[:1] == ['self']) else fn_argSpec.args
    if ('IN' not in  fn_args):
        assert False, '{}:need IN in args, but args={}'.format(name, fn_args)

    AX(fn_args, name + ':need all args has default (args.Len==defaults.Len)').is_length(len(fn_argSpec.defaults or tuple()))

    @functools.wraps(fn)
    def asTestFlowFn_wrapper(*args, **kwargs):
        try:
            IN = kwargs.get('IN') or DictObject(**kwargs)
            AX(IN, name+':IN').is_instance_of(DictObject)
            IN.__Name__ = name

            # 自动根据函数中的参数默认值定义，来更新参数的默认值, exam:
            # def fn(IN={}, item_type="", modFile="lambda: 'templet_{}.json'.format(IN.item_type)"):
            #     ...
            # AX(set(IN.keys()) - set(fn_args), name+':need set(IN.keys()) <= set(args)').is_length(0) #TODO 改为可以多传参
            for i in range(len(fn_args)): # 按顺序来，若有参数间依赖的需注意顺序
                arg = fn_args[i]
                default = fn_argSpec.defaults[i]
                if arg in ['IN']:
                    continue
                if isinstance(default, basestring) and default.startswith('lambda:'):
                    default_val = eval(default, dict(IN=IN, Libs=Libs.ENV, PC=Libs.PC, INI=Libs.INI, Utils=Utils))()
                    default_type = type(default_val)
                else:
                    default_val = default
                    # TODO: 参数default是一个类时，表示是必选参、又可以通过类方法__IsinstanceT__来更精确校验类型, exam: stage=PC.EStage
                    default_type = type(default) if not Utils.isSubCls(default) else default
                    if Utils.isSubCls(default):
                        default_type = default
                        if arg not in IN.keys():
                            assert False, 'miss MUST_ARG {} # when arg.default isa class'.format(arg)
                    else:
                        default_type = type(default)

                if IN.get(arg) is None:
                    IN[arg] = default_val

                if hasattr(default_type, '__ChkT__'):
                    default_type.__ChkT__(IN[arg], name+':need IN[{}] isa default_type={}'.format(arg, default_type))
                else:
                    AX(IN[arg], name+':need IN[{}] isa default_type={}'.format(arg, default_type)).\
                        doCalled(Utils.isinstanceT, default_type).is_true()
            # IN_kwargs = dict(**IN)
            # IN_kwargs.update(dict(IN=IN))
            IN_kwargs = TestFlow.IN_kwargs(IN)

            if fn_argSpec.args[0]=='self' and isinstance(args[0], TestBizKw):
                testBizKw = TYPE and TestBizKw() and args[0]
                if not testBizKw.__Inited__:
                    testBizKw.__Init__()
                    AX(testBizKw.__Inited__, '{}: self.__Inited__==True after self.__Init__()'.format(testBizKw)).is_true()

            argsNew = [] if not fn_argSpec.args[:1]==['self'] else args[:1]
            testFlow = fn(*argsNew, **IN_kwargs)
            AX(testFlow, name+':testFlow').is_instance_of(TestFlow)
            testFlow.run()
            ret = testFlow.toJson(**kwargs)
            if [__i for __i in inspect.stack() if __i[1].find('\\helpers\\pydev\\') != -1]: #PyDev调试模式
                ret['__testFlowObj__'] = testFlow
            return ret
        except Exception as errObj:
            Libs.logger.warn('fn={} errObj={}'.format(fn, errObj))
            return TestFlowFail(name='<asTestFlowFn>{}'.format(fn.__name__), failMsg=Utils.err_detail(errObj)).toJson(**kwargs)

    setattr(asTestFlowFn_wrapper, '__asTestFlowFn__', TestFlow)
    setattr(asTestFlowFn_wrapper, '__OriginArgspecStr__', Utils.fn_argspecStr(fn))
    return asTestFlowFn_wrapper


_asSingleton_SingletonStores = {}
def asSingleton(cls):

    inst = cls()
    assert cls.__name__ not in _asSingleton_SingletonStores.keys(),\
        "assert <cls.__name__={!r}> not in <_asSingleton_SingletonStores.keys()>".format(cls.__name__)
    _asSingleton_SingletonStores[cls.__name__] = cls
    inst.__SingletonName__ = cls.__name__
    #inst.__call__ = lambda: inst
    return inst




def defTestStep(case):
    """
        def fn(*args, **kwargs)
            pass
        #fn = defTestStep(my_step)
        dict_ = fn(*args, **kwargs)
        fn = TestStep(name=fn.__name__, **dict_) if dict_ != None else None
    """
    AX(case, 'case').is_instance_of(TestCase)
    case = TYPE and TestCase and case
    def defTestStep_decorator(fn):
        @functools.wraps(fn)
        def defTestStep_wrapper(*args, **kwargs):
            fn_argSpec = inspect.getargspec(fn)
            AX(fn_argSpec, '{}<argSpec>'.format(fn.__name__)).doAttrs(['args', 'varargs', 'keywords']). \
                isItemsEq(dict(args=['V', 'title'], varargs=None, keywords=None))
            AX(fn_argSpec, '{}<argSpec>'.format(fn.__name__)).doAttr('defaults').is_length(1).doGeti(0).is_instance_of(
                basestring)

            try:
                testStep = fn(*args, **kwargs)
            except Exception as errObj:
                pass
            if testStep != None:
                AX(testStep, 'testStep@{}'.format(fn.__name__)).is_instance_of(TestStep)
                testStep = TYPE and TestStep() and testStep
                testStep._pathSet(fn.__name__)
                testStep.title = fn_argSpec.defaults[0]

            return testStep
        case.addStepFn(defTestStep_wrapper)
        return defTestStep_wrapper
    return defTestStep_decorator


class TestStep(TestFlow):
    class Chks(TestStructAbstr):
        def __init__(self, **Chks):
            TestStructAbstr.__init__(self, path=str(self))

            self.Chks = DictObject()
            newChks = Chks['__Ordered__'] if Chks.get('__Ordered__') else Chks
            self.Chks.__OrderedKeys__ = newChks.keys()   # 如果是 OrderedDict，则这里会有顺序
            for name, respChk in newChks.items():
                assert isinstance(respChk, TestStep.Chk),\
                    'respChk={!r}: assert isinstance(respChk, TestStep.Chk)'.format(respChk)
                self.Chks[name] = respChk
                self.addSon(name, respChk)
        def toJson(self, **kwargs):
            return Libs.dictToJson(
                {k:v.toJson(**kwargs) for k,v in self.Chks.items()},
                self.path, **kwargs)
        def _isEmpty(self):
            return len(set(self.Chks.keys()) - set(['__OrderedKeys__'])) == 0


    class Chk(TestStructAbstr):
        def __init__(self, jp=None, expect=Libs.Undefined, extractor='',
                     chkExec='Chk.fact==Chk.expect if Chk.expect!=Libs.Undefined else Chk.fact!=Libs.Undefined'):
            TestStructAbstr.__init__(self, path=str(self))

            if jp == None:
                return
            self.jp = jp;
            assert isinstance(jp, basestring), "jp={!r}: assert isinstance(jp, basestring)".format(jp)
            self.chkExec = chkExec
            self.expect = expect
            self.fact = None
            self.extractor = extractor

        def toJson(self, **kwargs):
            return Libs.dictToJson(dict(
                jp=self.jp, chkExec=self.chkExec, expect=self.expect, extractor=self.extractor, fact=self.fact
            ), self.path, **kwargs)

    @property
    def RES(self):
        return self._RES
    @RES.setter
    def RES(self, value):
        # if not isinstance(value, (dict, Utils.JsonLike)):
        #     assert isinstance(value, (dict, Utils.JsonLike)),\
        #         'assert isinstance(value, (dict, Utils.JsonLike)), but value isa {}'.format(type(value))
        self._RES = Libs.dictToJson(value, self.path+'.RES')

    def _auto_lazyloadAttrs(self):
        lazyloads = [_ for _ in self.__can_lazyloadAttrs if isinstance(_, types.LambdaType)]
        if len(lazyloads) != 0:
            Libs.logDbg('TestStep({}): will auto self.attr=self.attr() in lazyloads={}'.format(self.name, lazyloads))
            for _lazyloadAttr in lazyloads:
                _lazyloadValue = getattr(self, _lazyloadAttr)
                setattr(_lazyloadAttr, _lazyloadValue())

    def __init__(self, reqUrl='', reqMethod='post', title='', name='step', reqJson={}, tryTimes=None, timeout=None,
                 Chks=UndefCls('Chks'), chksExt='', Post='', _caseObj=None, IN={}, **reqExtKw):
        super(TestStep, self).__init__(name=name, timeout=timeout)
        if reqUrl=='':  # just for (testStep = TYPE and TestStep())
            return

        self.__can_lazyloadAttrs = [
            'reqUrl', 'reqMethod', 'title', 'name', 'reqJson', 'Chks', 'chksExt', 'Post', 'reqExtKw', 'IN'
        ]
        tryTimes = tryTimes or Libs.INI.TryTimes
        self._caseObj = _caseObj
        self._pathPre = '{}'.format(Libs.name(reqUrl, name))

        self.title = title
        self.reqUrl = reqUrl
        if not (isinstance(reqUrl, basestring)
            or (Utils.isinstanceT(self.reqUrl, TestFlow.TestFlowLikeFn))
            or TestDriver.isaDvMethod(self.reqUrl)
            or TestBizKw.isaBizKwMethod(self.reqUrl)
        ):
            assert False, 'assert reqUrl={!r} isa str or TestFlow.TestFlowLikeFn or TestDriver.isaDvMethod(reqUrl)'.format(self.reqUrl)
        self.IN = IN       #TODO 一般只有当用于@routeTestFlow时才会用到，仅用于方便传递参数
        self.OUT = DictObject({})
        self.reqJson = reqJson
        self.reqMethod = reqMethod
        self.reqExtKw = {k: v for k, v in reqExtKw.items() if k[0].islower()}
        self.tryTimes = tryTimes
        self.timeout = (timeout if not reqJson.get('timeout') else reqJson['timeout']) or Libs.INI.Timeout
        self.Chks = TestStep.Chks() and Chks
        self.addSon('Chks', self.Chks)

        AX(chksExt, 'chksExt').is_instance_of(basestring)
        self.chksExt = chksExt
        self.RES = {}
        self.failMsg = ''
        self.Post = Post
        self._VStep = DictObject(Chks=self.Chks.Chks, RES=DictObject({}), IN=self.IN, OUT=self.OUT)
        self._hasRuned = False
        self._stacks = [__i for __i in inspect.stack() if __i[1].find('\\helpers\\pydev\\') == -1]

    def _beAddToCase(self, caseObj):
        self._caseObj = caseObj

    def run(self):
        if self._hasRuned:
            return

        self._hasRuned = True

        V = self._caseObj.V if self._caseObj else DictObject({self.name: self._VStep}) # 兼容@routeTestFlow形式下的单TestStep
        _VStep = self._VStep
        PC = Libs.PC     # Post中可能会用到

        Libs.logDbg('  {}: {}  #tryTimes={}'.format(self.path, self.title, self.tryTimes))
        self._auto_lazyloadAttrs()
        if self.Chks._isEmpty() and Utils.isinstanceT(self.reqUrl, TestFlow.TestFlowLikeFn):
            self.Chks = TestStep.Chks(   # 对Chks默认值为空、且目标为testFlow时
                code=TestStep.Chk(jp='$.code', expect=Libs.Code_Suc)
            )

        if isinstance(self.reqUrl, basestring):
            #or Utils.isinstanceT(self.reqUrl, TestFlow.TestFlowLikeFn)
            httpjson_ret = Libs.httpjson_post(reqMethod=self.reqMethod, reqUrl=self.reqUrl, reqJson=self.reqJson, reqExtKw=self.reqExtKw,
                                              tryTimes=self.tryTimes, timeout=self.timeout)
        elif TestDriver.isaDvMethod(self.reqUrl) or TestBizKw.isaBizKwMethod(self.reqUrl):
            RES = None
            errObj = None
            try:
                RES = self.reqUrl(*self.reqJson.get('__Args__', []), **self.reqJson)
            except Exception as errObj:
                return self.toReturn('call_reqUrl({})'.format(self.reqUrl), errObj)
            httpjson_ret = Libs.HttpjsonResp(RES=RES, errObj=errObj)
        else:
            assert False, 'assert reqUrl={!r} isa str or (TestDriver/TestBizKw)method'.format(self.reqUrl)

        # TODO: 这里一定要用DictObject不能用dict才能保证 _VStep.RES能传址引用
        self.RES = DictObject(httpjson_ret.RES) if isinstance(httpjson_ret.RES, dict) else httpjson_ret.RES
        self._VStep.RES = self.RES
        if httpjson_ret.errObj:
            return self.toReturn('errObj', Utils.err_detail(httpjson_ret.errObj))

        if self.RES.get('failMsg'):
            return self.toReturn('RES.failMsg', self.RES.get('failMsg'))

        for name in self.Chks.Chks.__OrderedKeys__:
            Chk = TYPE and TestStep.Chk() and self.Chks.Chks[name]
            try:
                assert isinstance(Chk.chkExec, basestring), 'assert isinstance(Chk.chkExec, basestring), but Chk.chkExec={!r}'.format(Chk.chkExec)
                Chk.fact = jpFinds(self.RES, Chk.jp, dftRaise=False, dft=Libs.Undefined)  # Libs.UndefinedStr才能被JSON化
                if Chk.extractor:
                    Chk.fact = eval(Chk.extractor.encode('utf-8'), dict(Chk=DictObject(fact=Chk.fact)))
            except Exception as errObj:
                return self.toReturn('Chks.{}({})'.format(name, Chk.chkExec), errObj)
            try:
                if not eval(Chk.chkExec.encode('utf-8'), dict(Chk=Chk, Chks=self.Chks.Chks, V=V, _VStep=_VStep, Libs=Libs, Utils=Utils)):
                    # failMsg = 'Chks.{}(chkExec={!r}), but fact={} expect={})'.format(name, Chk.chkExec, Chk.fact, Chk.expect)
                    failMsg = str_fmt('(but fact={} expect={})', json_o2t(Chk.fact), json_o2t(Chk.expect))
                    # if self.RES.get('failMsg'):
                    #     failMsg = '{} RES.failMsg=\n    {}'.format(failMsg, self.RES.get('failMsg'))
                    return self.toReturn('Chks.{}({})'.format(name, Chk.chkExec), failMsg)
            except Exception as errObj:
                return self.toReturn('Chks.{}({})'.format(name, Chk.chkExec), errObj)

        if self.chksExt and not eval(self.chksExt.encode('utf-8'), dict(Chks=self.Chks.Chks, V=V, _VStep=_VStep, Libs=Libs, Utils=Utils)):
            return self.toReturn('chksExt', self.chksExt)

        if self.Post:
            Post_cmds = self.Post.strip().split('\n')
            try:
                for Post_cmd in Post_cmds:
                    Libs.logDbg('    exec(Post): {}'.format(Post_cmd))
                    exec(Post_cmd.strip())
            except Exception as errObj:
                return self.toReturn('Post', Utils.err_detail(errObj))
            Libs.logDbg('  --> self.RES<{}> _VStep.RES<{}>'.format(id(self.RES), id(_VStep.RES)))
        return

    def toReturn(self, failKey=None, failMsg='', if_failStackAppend=True):
        if failMsg:
            istack = self._stacks[1]
            failStackAppend ='##nbtest_appendStack##\nFile "{}", line {}, in {}\n    {}\n##end##'.\
                format(istack[1], istack[2], istack[3], '\n'.join(istack[4]).strip('\n'))

            self.failMsg = Libs.failMsg(
                '{}.{}'.format(self.path, failKey),
                failMsg,
                failStackAppendLines=failStackAppend.split('\n')
            )
        return

    def toJson(self, **kwargs):
        result = Libs.dictToJson(dict(
            name=self.name, title=self.title, IN=self.IN, OUT=self.OUT,
            reqUrl=str(self.reqUrl), reqJson=self.reqJson, # 造数据时太长的reqJson无关注意义
            reqMethod=self.reqMethod, reqExtKw=self.reqExtKw,
            RES=Libs.dictToJson(self.RES, self.path+'.RES', **kwargs),
            Chks=self.Chks.toJson(**kwargs),
            chksExt=self.chksExt,
            Post=self.Post, failMsg=self.failMsg, code=Libs.Code_Err if self.failMsg else Libs.Code_Suc
        ), self.path+'.toJson()', **kwargs)
        return Libs.to_RES(result)

    def ifSuc(self):
        return not bool(self.failMsg)

class TestCase(TestFlow):
    """ TestCase由多个有序的TestStep组成，串行执行、滞后执行(step2可能依赖step1执行后产生的变量) """
    def __init__(self, name, IN, stepSleep=0, _pathPre='', timeout=None, **kwargs):
        super(TestCase, self).__init__(name=name, timeout=timeout)
        self.IN = DictObject(**IN)  # TODO: 当配套使用@routeCase时，如使用self.IN=IN 会出现传地址性关联，导致奇怪错误
        self.OUT = DictObject({})
        self.stepSleep = stepSleep
        assert isinstance(stepSleep, int), 'assert isinstance(stepSleep, int), stepSleep={!r}'.format(stepSleep)

        self.timeout = timeout or Libs.INI.Timeout
        self.stepFns = []
        self._stepObjs = []
        self.failMsg = ''
        self.code = 200
        self.V = DictObject(IN=self.IN, OUT=self.OUT)
        self._stacks = [__i for __i in inspect.stack() if __i[1].find('\\helpers\\pydev\\') == -1]

    def addStepFn(self, stepFn):
        assert isinstance(stepFn, types.FunctionType), 'step={!r}: assert isinstance(step, function)'.format(stepFn)
        # assert isinstance(stepFn, TestStep), 'step={!r}: assert isinstance(step, TestStep)'.format(stepFn)
        stepFn = TYPE and TestStep() and stepFn
        stepName = stepFn.__name__
        AX(stepFn, stepName).isNotIn([i.keys()[0] for i in self.stepFns])
        self.stepFns.append({stepName: stepFn})
        self.__updCaseTimeout(Libs.INI.TimeoutShort) #(stepFn.timeout)

    def __updCaseTimeout(self, timeout):
        self.timeout += timeout

    # def getStep(self, name):
    #     index = self._stepIndexs[name]
    #     return self._stepIndexs[index]
    def _runtimeAddStep(self, step):
        step = TYPE and TestStep() and step
        AX(step, '<title={}>step'.format(step.title)).doAttr('name').isNotIn(self.V.keys())
        self._stepObjs.append(step)
        self.V[step.name] = step._VStep
        step._beAddToCase(self)

        step_reqUrl_name = Libs.name(step.reqUrl, '')
        #step_path = 'steps.{}{}'.format(step.name, step_reqUrl_name and '<{}>'.format(step_reqUrl_name))
        self.addSon('steps.'+step.name, step)

    def toReturn(self, failKey=None, failMsg='', if_failStackAppend=True):
        if failMsg:
            istack = self._stacks[1]
            failStackAppend ='##nbtest_appendStack##\nFile "{}", line {}, in {}\n    {}\n##end##'.\
                format(istack[1], istack[2], istack[3], '\n'.join(istack[4]))

            self.failMsg = Libs.failMsg(
                '{}.{}'.format(self.path, failKey),
                failMsg,
                failStackAppendLines=failStackAppend.split('\n')
            )
        return
    def run(self):
        V = self.V
        i = 0
        step = None
        try:
            for i in range(len(self.stepFns)):
                stepFn = self.stepFns[i].values()[0]
                step = stepFn(V) #stepFn # stepFn(V)  TODO: step的滞后加载放在，改为放在TestStep内部去兼容
                if not step:
                    continue
                step = TYPE and TestStep() and step
                self._runtimeAddStep(step)

                step.run()
                if step.failMsg:
                    self.code = -(1000+i)
                    return self.toReturn(failKey=step.path, failMsg=step.failMsg)
                if self.stepSleep > 0:
                    time.sleep(self.stepSleep)
        except Exception as errObj:
            self.code = -(1000 + i)
            failKey = step.path if step else 'RES.steps[{}]'.format(i)
            return self.toReturn(failKey=failKey, failMsg=Utils.err_detail(errObj))
        return

    def getStepsBrief(self):
        return [dict(title='{}: {}'.format(step.name,step.title), result=step.ifSuc()) for step in self._stepObjs]

    def toJson(self, __YFLow__={}, **kwargs):
        assert isinstance(__YFLow__, dict), 'assert isinstance(__YFLow__, dict)'
        result = Libs.dictToJson(dict(
            name=self.name, steps=[step.toJson(**kwargs) for step in self._stepObjs],
            code=self.code, IN=self.IN, OUT=self.OUT, failMsg=self.failMsg
        ), self.path+'.toJson()', **kwargs)
        if len(__YFLow__.keys()):  # 适配 YFlow
            result["errno"] = 1 if result["code"]!=Libs.Code_Suc else 0
            result["data"] = self.getStepsBrief()
            result["errmsg"] = self.failMsg
            for k, v in __YFLow__.items():
                assert k not in result.keys(),\
                    "assert (__YFLow__)'s k={} not in {}".format(k, result.keys())
                result[k] = v
        return Libs.to_RES(result)

class TestSuite(TestFlow):
    """
        run TestCases use multi-thread
    """
    def __init__(self, name, timeout=None, ifRaiseTimeout=True):
        super(TestSuite, self).__init__(name=name, timeout=timeout)
        self._pathPre = ''
        self.failMsg = ''
        self.code = Libs.Code_Suc
        self.caseKwargs = {}
        self.sons = {}
        self.timeout = timeout or Libs.INI.Timeout
        self.ifRaiseTimeout = ifRaiseTimeout

    def _getPathToSon(self, son):
        return '{}.sons.{}'.format(self.path, son.name)

    def addCase(self, name, reqUrl='', reqJson={}, tryTimes=None, timeout=None):
        tryTimes = tryTimes or Libs.INI.TryTimes
        timeout = timeout or Libs.INI.Timeout
        assert isinstance(name, basestring), 'assert isinstance(name, basestring), name={!r}'.format(name)
        assert isinstance(reqJson, dict), 'assert isinstance(reqJson, dict), reqJson={!r}'.format(reqJson)
        if not isinstance(reqUrl, basestring):
            AX(reqUrl, 'when reqUrl not str').doCalled(Utils.isinstanceT, TestFlow.TestFlowLikeFn).is_true()
            #reqUrl = '{}/{}'.format(Libs.PC.Url_Ex, reqUrl.__name__) #TODO: 优化为reqUrl可以直接调用
            reqUrl = TYPE and TestFlow() and reqUrl
            timeout = max([timeout, reqUrl.timeout])
        elif not reqUrl:
            AX(name, 'when reqUrl not str').doMethod('find', '__').is_not_equal_to(-1)
            reqUrl = '{}/{}'.format(Libs.PC.Url_Ex, '_'.join(name.split('_')[:-2]))
        else:
            AX(reqUrl, 'when reqUrl not str').doMethod('find', '://').is_not_equal_to(-1)
            AX(reqUrl, 'when reqUrl not str').doGeti(-1).is_not_equal_to('/')

        fnName = reqUrl.split('/')[-1]
        if not name.startswith('{}__'.format(fnName)):
            name = '{}__{}'.format(fnName, name)
        AX(name, 'need name not exists').isNotIn(self.caseKwargs.keys())
        self.caseKwargs[name] = dict(
            reqUrl=reqUrl,
            reqJson=reqJson,
            tryTimes=tryTimes,
            timeout=timeout
        )

    def run(self):
        rets = Libs.threadsRun(Libs.httpjson_post, runOneKwargsDict=self.caseKwargs, name=self.name,
                               timeoutSum=self.timeout, ifRaiseTimeout=self.ifRaiseTimeout)

        failMsg = ""
        sons = {}
        for tcName, ret in rets.items():
            i_failMsg = ""
            ret = TYPE and Libs.HttpjsonResp() and ret
            if ret.errObj:
                i_failMsg = Libs.failMsg('', ret.errObj)
            else:
                i_failMsg = ret.RES.get('failMsg', '')
            sons[tcName] = ret.RES
            failMsg = (Libs.failMsg('sons.{}'.format(tcName), i_failMsg)) if i_failMsg else failMsg  # 用最新的非空i_failMsg替换公共的failMsg
        self.code = (-200 if failMsg else 200)
        self.failMsg = failMsg
        self.sons = sons
        return
    def toJson(self, **kwargs):
        result = Libs.dictToJson(dict(
            code=self.code, sons=self.sons, failMsg=self.failMsg
        ), self.path + '.toJson()', **kwargs)
        return Libs.to_RES(result)


class FlaskExt(Flask):
    def AutoHomeDocsStr(self, modules, reMatch='^(test|fn)\w+', path=''):
        docs_str = 'path={} reMatch={}'.format(path, reMatch)
        docs = {}

        for _r in self.url_map.iter_rules():
            k = _r.rule
            fn = self.view_functions[_r.endpoint]
            docs_str += '<h2>{}</h2>\n'.format(k)
            docs[k] = fn.__OriginArgspecStr__ if hasattr(fn, '__OriginArgspecStr__') else Utils.fn_argspecStr(fn)
            if fn.__doc__:
                docs[k] = "{}\n    '''\n    {}\n    '''".format(docs[k], fn.__doc__.strip())
            docs_str += '<p>{}\n</p>\n'.format(docs[k].replace('\n', '<br/>').replace(' ', '&nbsp;'))

        # AX(modules, 'modules').is_instance_of(dict)
        # for mName, m in modules.items():
        #     if not isinstance(m, types.ModuleType):
        #         continue
        #     fnDict = {
        #         i: getattr(m, i) for i in dir(m)
        #         if hasattr(getattr(m, i), '__RouteTestFlow__') and isinstance(getattr(m, i), types.FunctionType)
        #     }
        #     for k, fn in fnDict.items():
        #         docs_str += '<h2>/{}</h2>\n'.format(k)
        #         docs[k] = fn.__OriginArgspecStr__ if hasattr(fn, '__OriginArgspecStr__') else Utils.fn_argspecStr(fn)
        #         if fn.__doc__:
        #             docs[k] = "{}\n    '''\n    {}\n    '''".format(docs[k], fn.__doc__.strip())
        #         docs_str += '<p>{}\n</p>\n'.format(docs[k].replace('\n', '<br/>').replace(' ', '&nbsp;'))
        #
        #     instDict = {
        #         i: getattr(m, i) for i in dir(m)
        #         if isinstance(getattr(m, i), TestBizKw)
        #     }
        #     for instName, inst in instDict.items():
        #         methodDict = {
        #             i: getattr(inst, i) for i in dir(inst)
        #             if hasattr(getattr(inst, i), '__RouteTestFlow__') and isinstance(getattr(inst, i), types.MethodType)
        #         }
        #         print('AutoHomeDocsStr: {}::methodDict[{}].keys() = {}'
        #               .format(m.__name__, instName, {i: getattr(inst, i) for i in dir(inst) if i[:1] != '_'}))
        #         for k, fn in methodDict.items():
        #             docs_str += '<h2>/{}/{}</h2>\n'.format(instName, k)
        #             docs[k] = fn.__OriginArgspecStr__ if hasattr(fn, '__OriginArgspecStr__') else Utils.fn_argspecStr(
        #                 fn)
        #             if fn.__doc__:
        #                 docs[k] = "{}\n    '''\n    {}\n    '''".format(docs[k], fn.__doc__.strip())
        #             docs_str += '<p>{}\n</p>\n'.format(docs[k].replace('\n', '<br/>').replace(' ', '&nbsp;'))

        return docs_str

    @staticmethod
    def TempVar(**kwargs):
        return DictObject(**kwargs)


    @staticmethod
    def asTestFlow_Finder(m, app, logFn=print, ifInstFn=True, rulePre=''):
        """ 自动找出模块中满足 @asTestFlow* 装饰器的函数、类&方法，封装成api: /TestBizKw/asTestFlowFn """
        if rulePre:
            AX(rulePre, 'rulePre').is_instance_of(basestring).starts_with('/')

        AX(m, 'm').is_instance_of(types.ModuleType)
        AX(app, 'app').is_instance_of(FlaskExt)
        fnDict = {
            i: getattr(m, i) for i in dir(m)
            if hasattr(getattr(m, i), '__asTestFlowFn__') and isinstance(getattr(m, i), types.FunctionType)
        }
        for fnName, fn in fnDict.items():
            fn_asRouteTestFlow = app.routeTestFlow_asTestFlowFn('{}/{}'.format(rulePre, fnName))(fn)
            logFn('route({}): {}.{}'.format(fn_asRouteTestFlow.__FlaskExtEndpoint__, m.__name__, fnName))
            setattr(m, fnName, fn_asRouteTestFlow)

        if not ifInstFn:
            return

        instDict = {
            i: getattr(m, i) for i in dir(m)
            if isinstance(getattr(m, i), TestBizKw)
        }

        for instName, inst in instDict.items():
            methodDict = {
                i: getattr(inst, i) for i in dir(inst)
                if hasattr(getattr(inst, i), '__asTestFlowFn__') and isinstance(getattr(inst, i), types.MethodType)
            }
            logFn('asTestFlow_Finder: {}::methodDict[{}].keys() = {}'.format(m.__name__, instName, methodDict.keys()))
            for methodName, method in methodDict.items():
                method_asRouteTestFlow = app.routeTestFlow_asTestFlowFn('{}/{}/{}'.format(rulePre, instName, methodName))(method)
                logFn('route({}): {}.{}'.format(method_asRouteTestFlow.__FlaskExtEndpoint__, instName, method.__name__))
                setattr(inst, methodName, types.MethodType(method_asRouteTestFlow, inst))  # 注意这里不能直接赋值、而要绑为实例方法

    class RouteTestFlow(object):
        @classmethod
        def __IsinstanceT__(cls, o):
            return hasattr(o, '__RouteTestFlow__') and Utils.isSubCls(o.__RouteTestFlow__, TestFlow)


    def routeTestFlow_asTestFlowFn(self, rule, methods=['GET', 'POST'], **optionsOld):
        def routeTestFlow_asTestFlowFn_decorator(fn):
            options = dict(**optionsOld)
            options.update(dict(methods=methods))
            endpoint = options.pop('endpoint', None)

            ruleSplits = rule.strip('/').split('/')
            assert fn.__name__ == ruleSplits[-1],\
                'fn.__name__ == ruleSplits[-1], rule={}, fn={}'.format(rule, fn)
            name = fn.__name__
            fn_argSpec = inspect.getargspec(fn)
            @functools.wraps(fn)
            def routeTestFlow_asTestFlowFn_wrapper(*args, **kwargs):
                try:
                    Libs.logDbg('  -->{}(request={}, path={}, *{}, **{})'.format(name, request, request.path, args, kwargs))
                    if request.path == rule:
                        if request.method=='GET':
                            arg_Json = request.args.get('Json', '{}')
                            reqJson = json.loads(arg_Json)
                        else:
                            reqJson = request.json
                        IN = FlaskExt.TempVar(**reqJson)
                    else:  # route->fn, exam at TestStep.run()
                        IN = kwargs.get('IN') or DictObject(**kwargs)
                        AX(IN, name+':IN').is_instance_of(DictObject)
                    IN.__Name__ = name
                    # IN_kwargs = dict(**IN)
                    # IN_kwargs.update(dict(IN=IN))
                    IN_kwargs = TestFlow.IN_kwargs(IN)

                    argsNew = [] if not fn_argSpec.args[:1]==['self'] else args[:1]
                    result = fn(*argsNew, **IN_kwargs)
                    if IN_kwargs.get('__YFLow__'):
                        result.__YFLow__ = IN_kwargs
                    return jsonify(result) if (request.path == rule and not hasattr(fn, '__RouteTestFlow__')) else result
                except Exception as errObj:
                    result2 = dict(failMsg=Utils.err_detail(errObj))
                    return jsonify(result2) if (request.path == rule and not hasattr(fn, '__RouteTestFlow__')) else result2

            __FlaskExtEndpoint__ = rule #'__'.join(ruleSplits)
            try:
                if fn.__name__ in self.view_functions:
                    if not hasattr(self, '_BAK_view_functions'):
                        self._BAK_view_functions = {}
                    self._BAK_view_functions[fn.__name__] = self.view_functions.pop(fn.__name__)
                self.add_url_rule(rule, endpoint, routeTestFlow_asTestFlowFn_wrapper, **options)
            except Exception as errObj:
                raise errObj
            self.view_functions[__FlaskExtEndpoint__] = self.view_functions.pop(fn.__name__)
            for _r in self.url_map.iter_rules():
                if _r.rule != rule:
                    continue
                _r.endpoint = __FlaskExtEndpoint__
            self.url_map._rules_by_endpoint[__FlaskExtEndpoint__] = self.url_map._rules_by_endpoint.pop(fn.__name__)

            setattr(routeTestFlow_asTestFlowFn_wrapper, '__RouteTestFlow__', TestFlow)
            # setattr(routeTestFlow_asTestFlowFn_wrapper, '__asTestFlowFn__', TestFlow)
            setattr(routeTestFlow_asTestFlowFn_wrapper, '__OriginArgspecStr__',
                    Utils.fn_argspecStr(fn) if not hasattr(fn, '__OriginArgspecStr__') else fn.__OriginArgspecStr__
            )
            setattr(routeTestFlow_asTestFlowFn_wrapper, '__FlaskExtEndpoint__', __FlaskExtEndpoint__)
            return routeTestFlow_asTestFlowFn_wrapper
        return routeTestFlow_asTestFlowFn_decorator


    def routeTestFlow(self, rule, methods=['GET', 'POST'], **optionsOld):
        def routeTestFlow_decorator(fn):
            options = dict(**optionsOld)
            options.update(dict(methods=methods))
            endpoint = options.pop('endpoint', None)

            name = fn.__name__
            fn_argSpec = inspect.getargspec(fn)
            AX(fn_argSpec, name + ':fn_argSpec').doAttrs(['varargs', 'keywords']).\
                isItemsEq(dict(varargs=None, keywords="kwargs"))
            AX('IN', name+':need IN in args').isIn(fn_argSpec.args)
            AX(fn_argSpec.args, name + ':need len(args)==len(defaults)').is_length(len(fn_argSpec.defaults or tuple()))
            @functools.wraps(fn)
            def routeTestFlow_wrapper(*args, **kwargs):
                try:
                    Libs.logDbg('  -->{}(request={}, path={}, *{}, **{})'.format(name, request, request.path, args, kwargs))
                    if request.path == '/{}'.format(name):
                        if request.method=='GET':
                            arg_Json = request.args.get('Json', '{}')
                            reqJson = json.loads(arg_Json)
                        else:
                            reqJson = request.json
                        IN = FlaskExt.TempVar(**reqJson)
                    else:  # route->fn, exam at TestStep.run()
                        IN = kwargs.get('IN') or DictObject(**kwargs)
                        AX(IN, name+':IN').is_instance_of(DictObject)
                    IN.__Name__ = name

                    # 自动根据函数中的参数默认值定义，来更新参数的默认值, exam:
                    # def fn(IN={}, item_type="", modFile="lambda: 'templet_{}.json'.format(IN.item_type)"):
                    #     ...
                    # AX(set(IN.keys()) - set(fn_argSpec.args), name+':need set(IN.keys()) <= set(args)').is_length(0) #TODO 改为可以多传参
                    for i in range(len(fn_argSpec.args)): # 按顺序来，若有参数间依赖的需注意顺序
                        arg = fn_argSpec.args[i]
                        default = fn_argSpec.defaults[i]
                        if arg in ['IN']:
                            continue
                        if isinstance(default, basestring) and default.startswith('lambda:'):
                            default_val = eval(default, dict(IN=IN, ENV=Libs.ENV, PC=Libs.PC, Utils=Utils))()
                            default_type = type(default_val)
                        else:
                            default_val = default
                            # TODO: 参数default是一个类时，表示是必选参、又可以通过类方法__IsinstanceT__来更精确校验类型, exam: stage=Libs.PC.EStage
                            default_type = type(default) if not Utils.isSubCls(default) else default
                            if Utils.isSubCls(default):
                                default_type = default
                                if arg not in IN.keys():
                                    assert False, 'miss MUST_ARG {} # when arg.default isa class'.format(arg)
                            else:
                                default_type = type(default)

                        if IN.get(arg) is None:
                            IN[arg] = default_val

                        if hasattr(default_type, '__ChkT__'):
                            default_type.__ChkT__(IN[arg], name+':need IN[{}] isa default_type={}'.format(arg, default_type))
                        else:
                            AX(IN[arg], name+':need IN[{}] isa default_type={}'.format(arg, default_type)).\
                                doCalled(Utils.isinstanceT, default_type).is_true()
                    # IN_kwargs = dict(**IN)
                    # IN_kwargs.update(dict(IN=IN))
                    IN_kwargs = TestFlow.IN_kwargs(IN)

                    argsNew = [] if not fn_argSpec.args[:1]==['self'] else args[:1]
                    testFlow = fn(*argsNew, **IN_kwargs)
                    AX(testFlow, name+':testFlow').is_instance_of(TestFlow)
                    testFlow.run()
                    result = testFlow.toJson(__YFLow__=IN if IN.get('__YFLow__') else {})
                    return jsonify(result)
                except Exception as errObj:
                    return jsonify(dict(failMsg=Utils.err_detail(errObj)))

            self.add_url_rule(rule, endpoint, routeTestFlow_wrapper, **options)
            setattr(routeTestFlow_wrapper, '__RouteTestFlow__', TestFlow)
            setattr(routeTestFlow_wrapper, '__OriginArgspecStr__', Utils.fn_argspecStr(fn))
            return routeTestFlow_wrapper
        return routeTestFlow_decorator



if __name__ == '__main__':
    def _if_main():
        pass