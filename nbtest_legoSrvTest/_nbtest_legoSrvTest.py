# coding=utf-8
from __future__ import unicode_literals, print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json, types, requests, sys, codecs, traceback, threading, random, functools, inspect, re
import datetime, time   # time 模块不能删，用作eval('time.sleep')
from types import NoneType
from collections import OrderedDict
import logging


from DictObject import DictObject

from nbtest import Utils
from nbtest.Utils import Undef, TYPE
from nbtest.assertpyx import AX
from flask import Flask, request, jsonify, Response


class Libs(object):
    class PC(object):
        Url_Ex = "http://127.0.0.1:9901"
        Timeout = 30
        TryTimes = 1
    @classmethod
    def updPC(cls, pcCls):
        AX(pcCls, '').doCalled(Utils.isSubCls, cls.PC).is_true()
        cls.PC = pcCls
    class ENV(object):
        pass
    @classmethod
    def updENV(cls, envCls):
        AX(envCls, '').is_instance_of(Libs.ENV)
        cls.ENV = envCls

    Code_Suc = 200
    Code_Err = -200

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
    def toJson(**kwargs):
        retJson = {}
        for k, v in kwargs.items():
            assert isinstance(k, (NoneType, bool, int, long, float, basestring, list, dict)), \
                'isinstance(value, (NoneType, bool, int, long, float, basestring, list, dict)): {}={1!r}' \
                    .format(k, v)
            retJson[k] = v
        return retJson


    class HttpjsonResp(object):
        def __init__(self, RES=None, reqUrl=None, reqJson=None, errObj=None):
            self.reqUrl = reqUrl
            self.reqJson = reqJson
            self.RES = RES
            self.errObj = errObj

    @staticmethod
    def httpjson_post(reqUrl='', reqJson={}, reqMethod='post', reqExtKw={},
                      tryTimes=None, timeout=None, __ThreadRets__=None, __ThreadName__=None):
        _resp = None
        errObj = None
        timeout = timeout or Libs.PC.Timeout
        RES = None

        testFlowLike = Utils.isinstanceT(reqUrl, FlaskExt.RouteTestFlow)
        times = 1 if testFlowLike else (tryTimes or Libs.PC.TryTimes)
        for i_times in range(times):
            RES = None
            errObj = None
            try:
                Libs.logDbg('    --><{}>: {} timeout={} tryTimes=({}/{})'.format(
                    threading.current_thread().name, reqUrl, timeout, i_times, times
                ))
                if testFlowLike:
                    _resp = reqUrl(T=FlaskExt.TempVar(**reqJson))  # @FlaskExt.routeTestFlow()
                    _resp.text = _resp.data
                    assert _resp.status_code == 200, '<testFlowLike> status_code == 200, but status_code={}'.format(_resp.status_code)
                    RES = _resp.json
                else:
                    _resp = requests.request(method=reqMethod, url=reqUrl, json=reqJson, timeout=timeout, **reqExtKw)
                    assert _resp.status_code == 200, 'status_code == 200, but status_code={}'.format(_resp.status_code)
                    RES = _resp.json()
                break
            except Exception as errObj:
                if '{!r}'.format(errObj).find('timeout') != -1 or '{!r}'.format(errObj).find('status_code') != -1:
                    continue
                if _resp==None:
                    _resp = DictObject(status_code=None, text=None)
                RES = {
                    '__error_at_respObj__': dict(status_code=_resp.status_code, text=_resp.text, reqUrl=str(reqUrl))
                }
                break

        ret = Libs.HttpjsonResp(
            reqUrl=reqUrl, reqJson=reqJson, RES=RES, errObj=Exception(Utils.err_detail(errObj)) if errObj else None
        )
        if __ThreadRets__:
            __ThreadRets__.addRet(__ThreadName__ or '{}--{}'.format(reqUrl, id(_resp)), ret)
        return ret

    @staticmethod
    def failMsg(failKey='', failMsg=''):
        if (not failKey) and (not failMsg):
            return ''
        if isinstance(failMsg, Exception):
            return '{}: failMsg=\n{}'.format(failKey, Utils.err_detail(failMsg))
        else:
            return '{}: failMsg=\n{}'.format(failKey, failMsg)

    @staticmethod
    def to_RES(RES={}):
        assert isinstance(RES, dict), 'assert isinstance(RES, dict)'
        if RES.has_key('failMsg') and not RES.get('failMsg'):
            RES.pop('failMsg')
        if not RES.has_key('A'):
            if not RES.has_key('A_code') and RES.has_key('code'):
                RES['A_code'] = RES['code']
            if not RES.has_key('A_title') and RES.has_key('title'):
                RES['A_title'] = RES['title']
        return RES

    class ThreadRets(object):
        def __init__(self):
            self._rets = {}
        def addRet(self, key, val):
            assert isinstance(key, basestring),\
                'isinstance(key, basestring), but type(key)={}'.format(type(key))
            self._rets[key] = val
        def lenRets(self):
            return len(self._rets)
        def rets(self):
            return self._rets

    @staticmethod
    def threadsRun(runOneFn, runOneKwargsDict, timeoutSum=50, name='', ifRaiseTimeout=True, __ThreadGap__=0.5):
        """ exam:
        threadsRun(httpjson_post, runOneKwargsDict={
            "item_id_1": {reqUrl=Url_Ex+"/testAnItem_FastAudit",reqJson=dict(item_id="item_id_1",item_type="kVideoSet4Many")},
            "item_id_2": {reqUrl=Url_Ex+"/testAnItem_FastAudit",reqJson=dict(item_id="item_id_1",item_type="kVideoSet4Many")}
        })

        """
        __ThreadRets__ = Libs.ThreadRets()
        threads = {}
        _names = runOneKwargsDict.keys()
        _num = len(_names)
        for i in range(_num):
            _name = _names[i]
            runOneKwargs = runOneKwargsDict[_name]
            runOneKwargs.update(dict(__ThreadRets__=__ThreadRets__, __ThreadName__=_name, __ThreadGap__=__ThreadGap__*i))
            threads[_name] = threading.Thread(
                name = _name,
                target=runOneFn,
                kwargs=runOneKwargs
            )
        for _name, t in threads.items():
            t.setDaemon(True)
            t.start()

        _timeSum = timeoutSum+(__ThreadGap__*_num)
        _timeLogGap = int(_timeSum/100.0) or 1
        for sec in range(1, int(1+timeoutSum+(__ThreadGap__*_num))):
            if __ThreadRets__.lenRets()==len(threads) or (not len([t for note, t in threads.items() if t.isAlive()])):
                break
            time.sleep(1)
            if sec % 10 == 0:
                Libs.logDbg('--><{}>: threadsRun({}): ({}/<{}+{}*{}>)s'.format(threading.current_thread().name, name, sec, timeoutSum, __ThreadGap__, _num))
        else:
            if ifRaiseTimeout:
                raise Exception('threadsRun timeout')
        return __ThreadRets__.rets()


class CfgEnv(Libs.ENV):
    _CfgEnvs = DictObject({})
    @classmethod
    def loadCfgEnv(cls, envName):
        AX(envName, 'need envName in cfgEnvs.keys()').isIn(cls._CfgEnvs.keys())
        cfgEnv = cls._CfgEnvs[envName]
        Libs.updENV(cfgEnv)
        return TYPE and cls() and cfgEnv

    def __init__(self, __Name__=None, **kwargs):
        if __Name__ is None:
            return
        AX(__Name__, 'need __Name__ not exists in cfgEnvs.keys()').isNotIn(self.__class__._CfgEnvs.keys())
        self.__Name__ = __Name__
        self.__class__._CfgEnvs[__Name__] = self
        for k, v in kwargs.items():
            if k[0].isupper():
                setattr(self, k, v)


class TestDriver(object):
    @staticmethod
    def isaDvMethod(o):
        return hasattr(o, 'im_self') and isinstance(o.im_self, TestDriver)


class TestFlow(object):
    """ base_class for TestCase TestSuite TestModule """
    pass


def defTestStep(case):
    """
        def fn(*args, **kwargs)
            pass
        #fn = defTestStep(my_step)
        dict_ = fn(*args, **kwargs)
        fn = TestStep(name=fn.__name__, **dict_) if dict_ != None else None
    """
    AX(case, 'case').is_instance_of(TestCase)
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper_defTestStep(*args, **kwargs):
            fn_argSpec = inspect.getargspec(fn)
            AX(fn_argSpec, '{}<argSpec>'.format(fn.__name__)).doAttrs(['args', 'varargs', 'keywords']). \
                isItemsEq(dict(args=['V', 'title'], varargs=None, keywords=None))
            AX(fn_argSpec, '{}<argSpec>'.format(fn.__name__)).doAttr('defaults').is_length(1).doGeti(0).is_instance_of(
                basestring)

            testStep = fn(*args, **kwargs)
            if testStep != None:
                AX(testStep, 'testStep@{}'.format(fn.__name__)).is_instance_of(TestStep)
                testStep.name = fn.__name__
                testStep.title = fn_argSpec.defaults[0]
            return testStep
        case.addStepFn(wrapper_defTestStep)
        return wrapper_defTestStep
    return decorator

class TestStep(TestFlow):
    class Chks(object):
        def __init__(self, **Chks):
            if Chks == {}:
                return
            self.Chks = DictObject()
            newChks = Chks['__Ordered__'] if Chks.get('__Ordered__') else Chks
            self.Chks.__OrderedKeys__ = newChks.keys()   # 如果是 OrderedDict，则这里会有顺序
            for name, respChk in newChks.items():
                assert isinstance(respChk, TestStep.Chk),\
                    'respChk={!r}: assert isinstance(respChk, TestStep.Chk)'.format(respChk)
                self.Chks[name] = respChk
        def toJson(self, **kwargs):
            return Libs.toJson(**{k:v.toJson() for k,v in self.Chks.items()})

    class Chk(object):
        def __init__(self, jp=None, expect=None, chkExec='Chk.fact==Chk.expect'):
            if jp == None:
                return
            # self.name = name; assert isinstance(name, unicode), "name={!r}: assert isinstance(name, unicode)".format(name)
            self.jp = jp; assert isinstance(jp, unicode), "jp={!r}: assert isinstance(jp, unicode)".format(jp)
            self.chkExec = chkExec
            self.expect = expect
            self.fact = None
        def toJson(self, **kwargs):
            return Libs.toJson(jp=self.jp, chkExec=self.chkExec, expect=self.expect, fact=self.fact)


    def __init__(self, reqUrl='', reqMethod='post', title='', name=None, reqJson=None, tryTimes=2, timeout=None,
                 Chks=None, chksExt='', Post='', reqExtKw={}, _caseObj=None, IN={}):
        if not reqUrl:
            return
        self._caseObj = _caseObj
        self.name = name
        self._pathPre = ''

        self.title = title
        self.reqUrl = reqUrl
        if not (isinstance(reqUrl, basestring)
                or Utils.isinstanceT(self.reqUrl, FlaskExt.RouteTestFlow)
                or TestDriver.isaDvMethod(self.reqUrl)):
            assert False, 'assert reqUrl={!r} isa str or FlaskExt.RouteTestFlow or TestDriver.isaDvMethod(reqUrl)'.format(self.reqUrl)
        self.IN = IN       #TODO 一般只有当用于@routeTestFlow时才会用到，仅用于方便传递参数
        self.reqJson = reqJson
        self.reqMethod = reqMethod
        self.reqExtKw = reqExtKw
        self.tryTimes = tryTimes
        self.timeout = timeout
        self.Chks = TestStep.Chks() and Chks
        AX(chksExt, 'chksExt').is_instance_of(basestring)
        self.chksExt = chksExt
        self.RES = None
        self.failMsg = ''
        self.Post = Post
        self._VStep = DictObject(Chks=self.Chks.Chks, RES=DictObject({}), IN=self.IN)
        self._hasRuned = False

    def _beAddToCase(self, caseObj):
        self._caseObj = caseObj
        self._pathPre = caseObj._getPathToSonStep(self)

    def run(self):
        if self._hasRuned:
            return
        self._hasRuned = True

        V = self._caseObj.V if self._caseObj else DictObject({self.name: self._VStep}) # 兼容@routeTestFlow形式下的单TestStep
        _VStep = self._VStep
        PC = Libs.PC     # Post中可能会用到

        Libs.logDbg('  {}> {}  #tryTimes={}'.format(self.getPath(), self.title, self.tryTimes))

        if isinstance(self.reqUrl, basestring) or Utils.isinstanceT(self.reqUrl, FlaskExt.RouteTestFlow):
            httpjson_ret = Libs.httpjson_post(reqMethod=self.reqMethod, reqUrl=self.reqUrl, reqJson=self.reqJson, reqExtKw=self.reqExtKw,
                                              tryTimes=self.tryTimes, timeout=self.timeout)
        elif TestDriver.isaDvMethod(self.reqUrl):
            RES = None
            errObj = None
            try:
                RES = AX(self, self.name).doMethod('reqUrl', *self.reqJson.get('__Args__',[]), **self.reqJson).val
            except Exception as errObj:
                pass
            httpjson_ret = DictObject(RES=RES, errObj=errObj)
        else:
            assert False, 'assert reqUrl={!r} isa str or FlaskExt.RouteTestFlow or Utils.isFn(reqUrl)'.format(self.reqUrl)

        if httpjson_ret.errObj:
            return self.toReturn('RES', Utils.err_detail(httpjson_ret.errObj))
        self.RES = DictObject(httpjson_ret.RES)  # TODO: 这里一定要用DictObject不能用dict才能保证 _VStep.RES能传址引用
        self._VStep.RES = self.RES

        for name in self.Chks.Chks.__OrderedKeys__:
            Chk = TYPE and TestStep.Chk() and self.Chks.Chks[name]
            try:
                assert isinstance(Chk.chkExec, unicode), 'assert isinstance(Chk.chkExec, unicode), but Chk.chkExec={!r}'.format(Chk.chkExec)
                Chk.fact = Utils.jpFinds(self.RES, Chk.jp, dftRaise=False, dft=str(Undef))  # str(Undef)才能被JSON化
            except Exception as errObj:
                return self.toReturn('Chks.{}'.format(name), errObj)
            try:
                if not eval(Chk.chkExec.encode('utf-8'), dict(Chk=Chk, Chks=self.Chks.Chks, V=V, _VStep=_VStep)):
                    failMsg = '(expect ({}), but fact={} expect={})'.format(Chk.chkExec, Chk.fact, Chk.expect)
                    if self.RES.get('failMsg'):
                        failMsg = '{} RES.failMsg=\n    {}'.format(failMsg, self.RES.get('failMsg'))
                    return self.toReturn('Chks.{}'.format(name), failMsg)
            except Exception as errObj:
                return self.toReturn('Chks.{}(chkExec={!r})'.format(name, Chk.chkExec), errObj)

        if self.chksExt and not eval(self.chksExt.encode('utf-8'), dict(Chks=self.Chks.Chks, V=V, _VStep=_VStep)):
            return self.toReturn('chksExt', self.chksExt)

        if self.Post:
            Post_cmds = self.Post.strip().split('\n')
            # try:
            for Post_cmd in Post_cmds:
                Libs.logDbg('    exec(Post): {}'.format(Post_cmd))
                exec(Post_cmd.strip())
            # except Exception as errObj:
                # return self.toReturn('Post', Utils.err_detail(errObj))
            Libs.logDbg('  --> self.RES<{}> _VStep.RES<{}>'.format(id(self.RES), id(_VStep.RES)))
        return

    def toReturn(self, failKey=None, failMsg=''):
        if failMsg:
            self.failMsg = Libs.failMsg('{}.{}'.format(self.getPath(), failKey), failMsg)
        return

    def toJson(self, **kwargs):
        result = Libs.toJson(
            name=self.name, title=self.title, IN=self.IN,
            reqUrl=str(self.reqUrl), reqJson=self.reqJson, # 造数据时太长的reqJson无关注意义
            reqMethod=self.reqMethod, reqExtKw=self.reqExtKw,
            RES=self.RES, Chks=self.Chks.toJson(), chksExt=self.chksExt,
            Post=self.Post, failMsg=self.failMsg, code=Libs.Code_Err if self.failMsg else Libs.Code_Suc
        )
        return Libs.to_RES(result)

    def ifSuc(self):
        return not bool(self.failMsg)

    def getPath(self):
        return self._pathPre

class TestCase(TestFlow):
    """ TestCase由多个有序的TestStep组成，串行执行、滞后执行(step2可能依赖step1执行后产生的变量) """
    Timeout = Libs.PC.Timeout*Libs.PC.TryTimes*3
    def __init__(self, name, IN, stepSleep=0, _pathPre='', **kwargs):
        self.IN = DictObject(**IN)  # TODO: 当配套使用@routeCase时，如使用self.IN=IN 会出现传地址性关联，导致奇怪错误
        self.OUT = DictObject({})
        self.stepSleep = stepSleep
        assert isinstance(stepSleep, int), 'assert isinstance(stepSleep, int), stepSleep={!r}'.format(stepSleep)

        self.stepFns = []
        self._stepObjs = []
        self.name = name
        self.failMsg = ''
        self.code = 200
        self.V = DictObject(IN=self.IN, OUT=self.OUT)
        self._pathPre = _pathPre

    def addStepFn(self, stepFn):
        assert isinstance(stepFn, types.FunctionType), 'step={!r}: assert isinstance(step, function)'.format(stepFn)
        AX(stepFn, stepFn.__name__).isNotIn([i.keys()[0] for i in self.stepFns])
        self.stepFns.append({stepFn.__name__: stepFn})
    def getPath(self):
        return self._pathPre

    # def getStep(self, name):
    #     index = self._stepIndexs[name]
    #     return self._stepIndexs[index]
    def _runtimeAddStep(self, step):
        step = TYPE and TestStep() and step
        AX(step, '<title={}>step'.format(step.title)).doAttr('name').isNotIn(self.V.keys())
        self._stepObjs.append(step)
        self.V[step.name] = step._VStep
        step._beAddToCase(self)

    def _getPathToSonStep(self, step):
        return '{}.steps[?(@.name==\'{}\')]'.format(self.getPath(), step.name)

    def toReturn(self, failKey, failMsg):
        if failMsg:
            self.failMsg = Libs.failMsg('{}.{}'.format(self.getPath(), failKey), failMsg)
        return
    def run(self):
        V = self.V
        i = 0
        step = None
        try:
            for i in range(len(self.stepFns)):
                stepFn = self.stepFns[i].values()[0]
                step = stepFn(V)
                if not step:
                    continue
                step = TYPE and TestStep() and step
                self._runtimeAddStep(step)

                step.run()
                if step.failMsg:
                    self.code = -(1000+i)
                    return self.toReturn(failKey=self._getPathToSonStep(step), failMsg=step.failMsg)
                if self.stepSleep > 0:
                    time.sleep(self.stepSleep)
        except Exception as errObj:
            self.code = -(1000 + i)
            failKey = self._getPathToSonStep(step) if step else 'RES.steps[{}]'.format(i)
            return self.toReturn(failKey=failKey, failMsg=Utils.err_detail(errObj))
        return

    def getStepsBrief(self):
        return [dict(title='{}: {}'.format(step.name,step.title), result=step.ifSuc()) for step in self._stepObjs]

    def toJson(self, __YFLow__={}, **kwargs):
        assert isinstance(__YFLow__, dict), 'assert isinstance(__YFLow__, dict)'
        result = Libs.toJson(name=self.name, steps=[step.toJson() for step in self._stepObjs],
                    code=self.code, IN=self.IN, OUT=self.OUT, failMsg=self.failMsg)
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
    Timeout = TestCase.Timeout
    def __init__(self, name, timeout_suite=TestCase.Timeout, ifRaiseTimeout=True):
        self.name = name
        self._pathPre = ''
        self.failMsg = ''
        self.code = 200
        self.caseKwargs = {}
        self.sons = {}
        self.timeout_suite = timeout_suite
        self.ifRaiseTimeout = ifRaiseTimeout

    def getPath(self):
        return self._pathPre

    def _getPathToSon(self, son):
        return '{}.sons.{}'.format(self.getPath(), son.name)

    def addCase(self, name, reqUrl='', reqJson={}, tryTimes=1, timeout=TestCase.Timeout):
        assert isinstance(name, basestring), 'assert isinstance(name, basestring), name={!r}'.format(name)
        assert isinstance(reqJson, dict), 'assert isinstance(reqJson, dict), reqJson={!r}'.format(reqJson)
        if not isinstance(reqUrl, basestring):
            AX(reqUrl, 'when reqUrl not str').doCalled(Utils.isinstanceT, FlaskExt.RouteTestFlow).is_true()
            reqUrl = '{}/{}'.format(Libs.PC.Url_Ex, reqUrl.__name__)
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
        fn = Libs.httpjson_post

        rets = Libs.threadsRun(fn, runOneKwargsDict=self.caseKwargs, name=self.name,
                               timeoutSum=self.timeout_suite, ifRaiseTimeout=self.ifRaiseTimeout)

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
        result = Libs.toJson(code=self.code, sons=self.sons, failMsg=self.failMsg)
        return Libs.to_RES(result)


class FlaskExt(Flask):
    @classmethod
    def AutoHomeDocsStr(cls, globals_, reMatch='^(test|fn)\w+', path=''):
        docs_str = 'path={} reMatch={}'.format(path, reMatch)
        docs = {}
        for k, fn in globals_.items():
            if not ( Utils.isFn(fn) and hasattr(fn, '__RouteTestFlow__') and re.match(reMatch, k) ):
                continue
            docs_str += '<h2>{}</h2>\n'.format(k)
            docs[k] = fn.__OriginArgspecStr__ if hasattr(fn, '__OriginArgspecStr__') else Utils.fn_argspecStr(fn)
            if fn.__doc__:
                docs[k] = "{}\n    '''\n    {}\n    '''".format(docs[k], fn.__doc__.strip())
            docs_str += '<p>{}\n</p>\n'.format(docs[k].replace('\n', '<br/>').replace(' ', '&nbsp;'))

        return docs_str

    @staticmethod
    def TempVar(**kwargs):
        return DictObject(**kwargs)

    class RouteTestFlow(object):
        @classmethod
        def __IsinstanceT__(cls, o):
            return hasattr(o, '__RouteTestFlow__') and Utils.isSubCls(o.__RouteTestFlow__, TestFlow)

    def routeTestFlow(self, rule, **options):
        def decorator(fn):
            endpoint = options.pop('endpoint', None)

            name = fn.__name__
            fn_argSpec = inspect.getargspec(fn)
            AX(fn_argSpec, name + ':fn_argSpec').doAttrs(['varargs', 'keywords']).\
                isItemsEq(dict(varargs=None, keywords="kwargs"))
            AX('T', name+':need T in args').isIn(fn_argSpec.args)
            AX(fn_argSpec.args, name + ':need len(args)==len(defaults)').is_length(len(fn_argSpec.defaults or tuple()))
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    Libs.logDbg('  -->{}(request={}, path={}, *{}, **{})'.format(name, request, request.path, args, kwargs))
                    if request.path == '/{}'.format(name):
                        if request.method=='GET':
                            arg_Json = request.args.get('Json', '{}')
                            reqJson = json.loads(arg_Json)
                        else:
                            reqJson = request.json
                        T = FlaskExt.TempVar(**reqJson)
                    else:  # route->fn, exam at TestStep.run()
                        T = kwargs.get('T')
                        AX(T, name+':T').is_instance_of(DictObject)
                    T.__Name__ = name

                    # 自动根据函数中的参数默认值定义，来更新参数的默认值, exam:
                    # def fn(T={}, item_type="", modFile="lambda: 'templet_{}.json'.format(T.item_type)"):
                    #     ...
                    # AX(set(T.keys()) - set(fn_argSpec.args), name+':need set(T.keys()) <= set(args)').is_length(0) #TODO 改为可以多传参
                    for i in range(len(fn_argSpec.args)): # 按顺序来，若有参数间依赖的需注意顺序
                        arg = fn_argSpec.args[i]
                        default = fn_argSpec.defaults[i]
                        if arg == 'T':
                            continue
                        if isinstance(default, basestring) and default.startswith('lambda:'):
                            default_val = eval(default, dict(T=T, ENV=Libs.ENV, PC=Libs.PC, Utils=Utils))()
                            default_type = type(default_val)
                        else:
                            default_val = default
                            # TODO: 参数default是一个类时，表示是必选参、又可以通过类方法__IsinstanceT__来更精确校验类型, exam: stage=PC.EStage
                            default_type = type(default) if not Utils.isSubCls(default) else default
                            if Utils.isSubCls(default):
                                default_type = default
                                if arg not in T.keys():
                                    assert False, 'miss MUST_ARG {} # when arg.default isa class'.format(arg)
                            else:
                                default_type = type(default)

                        if T.get(arg) is None:
                            T[arg] = default_val

                        if hasattr(default_type, '__ChkT__'):
                            default_type.__ChkT__(T[arg], name+':need T[{}] isa default_type={}'.format(arg, default_type))
                        else:
                            AX(T[arg], name+':need T[{}] isa default_type={}'.format(arg, default_type)).\
                                doCalled(Utils.isinstanceT, default_type).is_true()

                    testFlow = fn(*args, T=T, **T)
                    AX(testFlow, name+':testFlow').is_instance_of(TestFlow)
                    testFlow.run()
                    result = testFlow.toJson(__YFLow__=T if T.get('__YFLow__') else {})
                    return jsonify(result)
                except Exception as errObj:
                    return jsonify(dict(failMsg=Utils.err_detail(errObj)))

            self.add_url_rule(rule, endpoint, wrapper, **options)
            setattr(wrapper, '__RouteTestFlow__', TestFlow)
            setattr(wrapper, '__OriginArgspecStr__', Utils.fn_argspecStr(fn))
            return wrapper
        return decorator


if __name__ == '__main__':
    def _if_main():
        pass