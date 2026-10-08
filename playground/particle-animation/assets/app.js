(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))r(a);new MutationObserver(a=>{for(const l of a)if(l.type==="childList")for(const d of l.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&r(d)}).observe(document,{childList:!0,subtree:!0});function t(a){const l={};return a.integrity&&(l.integrity=a.integrity),a.referrerPolicy&&(l.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?l.credentials="include":a.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function r(a){if(a.ep)return;a.ep=!0;const l=t(a);fetch(a.href,l)}})();var Vo=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function OT(s){return s&&s.__esModule&&Object.prototype.hasOwnProperty.call(s,"default")?s.default:s}var Fc={exports:{}},Go={},Oc={exports:{}},gt={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var jp;function T0(){if(jp)return gt;jp=1;var s=Symbol.for("react.element"),e=Symbol.for("react.portal"),t=Symbol.for("react.fragment"),r=Symbol.for("react.strict_mode"),a=Symbol.for("react.profiler"),l=Symbol.for("react.provider"),d=Symbol.for("react.context"),m=Symbol.for("react.forward_ref"),g=Symbol.for("react.suspense"),_=Symbol.for("react.memo"),M=Symbol.for("react.lazy"),u=Symbol.iterator;function f(U){return U===null||typeof U!="object"?null:(U=u&&U[u]||U["@@iterator"],typeof U=="function"?U:null)}var p={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},y=Object.assign,E={};function S(U,K,Le){this.props=U,this.context=K,this.refs=E,this.updater=Le||p}S.prototype.isReactComponent={},S.prototype.setState=function(U,K){if(typeof U!="object"&&typeof U!="function"&&U!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,U,K,"setState")},S.prototype.forceUpdate=function(U){this.updater.enqueueForceUpdate(this,U,"forceUpdate")};function v(){}v.prototype=S.prototype;function A(U,K,Le){this.props=U,this.context=K,this.refs=E,this.updater=Le||p}var P=A.prototype=new v;P.constructor=A,y(P,S.prototype),P.isPureReactComponent=!0;var L=Array.isArray,z=Object.prototype.hasOwnProperty,D={current:null},F={key:!0,ref:!0,__self:!0,__source:!0};function R(U,K,Le){var De,we={},se=null,_e=null;if(K!=null)for(De in K.ref!==void 0&&(_e=K.ref),K.key!==void 0&&(se=""+K.key),K)z.call(K,De)&&!F.hasOwnProperty(De)&&(we[De]=K[De]);var de=arguments.length-2;if(de===1)we.children=Le;else if(1<de){for(var Ie=Array(de),je=0;je<de;je++)Ie[je]=arguments[je+2];we.children=Ie}if(U&&U.defaultProps)for(De in de=U.defaultProps,de)we[De]===void 0&&(we[De]=de[De]);return{$$typeof:s,type:U,key:se,ref:_e,props:we,_owner:D.current}}function I(U,K){return{$$typeof:s,type:U.type,key:K,ref:U.ref,props:U.props,_owner:U._owner}}function W(U){return typeof U=="object"&&U!==null&&U.$$typeof===s}function O(U){var K={"=":"=0",":":"=2"};return"$"+U.replace(/[=:]/g,function(Le){return K[Le]})}var j=/\/+/g;function re(U,K){return typeof U=="object"&&U!==null&&U.key!=null?O(""+U.key):K.toString(36)}function ae(U,K,Le,De,we){var se=typeof U;(se==="undefined"||se==="boolean")&&(U=null);var _e=!1;if(U===null)_e=!0;else switch(se){case"string":case"number":_e=!0;break;case"object":switch(U.$$typeof){case s:case e:_e=!0}}if(_e)return _e=U,we=we(_e),U=De===""?"."+re(_e,0):De,L(we)?(Le="",U!=null&&(Le=U.replace(j,"$&/")+"/"),ae(we,K,Le,"",function(je){return je})):we!=null&&(W(we)&&(we=I(we,Le+(!we.key||_e&&_e.key===we.key?"":(""+we.key).replace(j,"$&/")+"/")+U)),K.push(we)),1;if(_e=0,De=De===""?".":De+":",L(U))for(var de=0;de<U.length;de++){se=U[de];var Ie=De+re(se,de);_e+=ae(se,K,Le,Ie,we)}else if(Ie=f(U),typeof Ie=="function")for(U=Ie.call(U),de=0;!(se=U.next()).done;)se=se.value,Ie=De+re(se,de++),_e+=ae(se,K,Le,Ie,we);else if(se==="object")throw K=String(U),Error("Objects are not valid as a React child (found: "+(K==="[object Object]"?"object with keys {"+Object.keys(U).join(", ")+"}":K)+"). If you meant to render a collection of children, use an array instead.");return _e}function X(U,K,Le){if(U==null)return U;var De=[],we=0;return ae(U,De,"","",function(se){return K.call(Le,se,we++)}),De}function Z(U){if(U._status===-1){var K=U._result;K=K(),K.then(function(Le){(U._status===0||U._status===-1)&&(U._status=1,U._result=Le)},function(Le){(U._status===0||U._status===-1)&&(U._status=2,U._result=Le)}),U._status===-1&&(U._status=0,U._result=K)}if(U._status===1)return U._result.default;throw U._result}var q={current:null},G={transition:null},J={ReactCurrentDispatcher:q,ReactCurrentBatchConfig:G,ReactCurrentOwner:D};function ie(){throw Error("act(...) is not supported in production builds of React.")}return gt.Children={map:X,forEach:function(U,K,Le){X(U,function(){K.apply(this,arguments)},Le)},count:function(U){var K=0;return X(U,function(){K++}),K},toArray:function(U){return X(U,function(K){return K})||[]},only:function(U){if(!W(U))throw Error("React.Children.only expected to receive a single React element child.");return U}},gt.Component=S,gt.Fragment=t,gt.Profiler=a,gt.PureComponent=A,gt.StrictMode=r,gt.Suspense=g,gt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=J,gt.act=ie,gt.cloneElement=function(U,K,Le){if(U==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+U+".");var De=y({},U.props),we=U.key,se=U.ref,_e=U._owner;if(K!=null){if(K.ref!==void 0&&(se=K.ref,_e=D.current),K.key!==void 0&&(we=""+K.key),U.type&&U.type.defaultProps)var de=U.type.defaultProps;for(Ie in K)z.call(K,Ie)&&!F.hasOwnProperty(Ie)&&(De[Ie]=K[Ie]===void 0&&de!==void 0?de[Ie]:K[Ie])}var Ie=arguments.length-2;if(Ie===1)De.children=Le;else if(1<Ie){de=Array(Ie);for(var je=0;je<Ie;je++)de[je]=arguments[je+2];De.children=de}return{$$typeof:s,type:U.type,key:we,ref:se,props:De,_owner:_e}},gt.createContext=function(U){return U={$$typeof:d,_currentValue:U,_currentValue2:U,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},U.Provider={$$typeof:l,_context:U},U.Consumer=U},gt.createElement=R,gt.createFactory=function(U){var K=R.bind(null,U);return K.type=U,K},gt.createRef=function(){return{current:null}},gt.forwardRef=function(U){return{$$typeof:m,render:U}},gt.isValidElement=W,gt.lazy=function(U){return{$$typeof:M,_payload:{_status:-1,_result:U},_init:Z}},gt.memo=function(U,K){return{$$typeof:_,type:U,compare:K===void 0?null:K}},gt.startTransition=function(U){var K=G.transition;G.transition={};try{U()}finally{G.transition=K}},gt.unstable_act=ie,gt.useCallback=function(U,K){return q.current.useCallback(U,K)},gt.useContext=function(U){return q.current.useContext(U)},gt.useDebugValue=function(){},gt.useDeferredValue=function(U){return q.current.useDeferredValue(U)},gt.useEffect=function(U,K){return q.current.useEffect(U,K)},gt.useId=function(){return q.current.useId()},gt.useImperativeHandle=function(U,K,Le){return q.current.useImperativeHandle(U,K,Le)},gt.useInsertionEffect=function(U,K){return q.current.useInsertionEffect(U,K)},gt.useLayoutEffect=function(U,K){return q.current.useLayoutEffect(U,K)},gt.useMemo=function(U,K){return q.current.useMemo(U,K)},gt.useReducer=function(U,K,Le){return q.current.useReducer(U,K,Le)},gt.useRef=function(U){return q.current.useRef(U)},gt.useState=function(U){return q.current.useState(U)},gt.useSyncExternalStore=function(U,K,Le){return q.current.useSyncExternalStore(U,K,Le)},gt.useTransition=function(){return q.current.useTransition()},gt.version="18.3.1",gt}var Kp;function hd(){return Kp||(Kp=1,Oc.exports=T0()),Oc.exports}/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var $p;function w0(){if($p)return Go;$p=1;var s=hd(),e=Symbol.for("react.element"),t=Symbol.for("react.fragment"),r=Object.prototype.hasOwnProperty,a=s.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,l={key:!0,ref:!0,__self:!0,__source:!0};function d(m,g,_){var M,u={},f=null,p=null;_!==void 0&&(f=""+_),g.key!==void 0&&(f=""+g.key),g.ref!==void 0&&(p=g.ref);for(M in g)r.call(g,M)&&!l.hasOwnProperty(M)&&(u[M]=g[M]);if(m&&m.defaultProps)for(M in g=m.defaultProps,g)u[M]===void 0&&(u[M]=g[M]);return{$$typeof:e,type:m,key:f,ref:p,props:u,_owner:a.current}}return Go.Fragment=t,Go.jsx=d,Go.jsxs=d,Go}var Zp;function A0(){return Zp||(Zp=1,Fc.exports=w0()),Fc.exports}var ut=A0(),it=hd(),fl={},Bc={exports:{}},Bn={},kc={exports:{}},zc={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Qp;function R0(){return Qp||(Qp=1,(function(s){function e(G,J){var ie=G.length;G.push(J);e:for(;0<ie;){var U=ie-1>>>1,K=G[U];if(0<a(K,J))G[U]=J,G[ie]=K,ie=U;else break e}}function t(G){return G.length===0?null:G[0]}function r(G){if(G.length===0)return null;var J=G[0],ie=G.pop();if(ie!==J){G[0]=ie;e:for(var U=0,K=G.length,Le=K>>>1;U<Le;){var De=2*(U+1)-1,we=G[De],se=De+1,_e=G[se];if(0>a(we,ie))se<K&&0>a(_e,we)?(G[U]=_e,G[se]=ie,U=se):(G[U]=we,G[De]=ie,U=De);else if(se<K&&0>a(_e,ie))G[U]=_e,G[se]=ie,U=se;else break e}}return J}function a(G,J){var ie=G.sortIndex-J.sortIndex;return ie!==0?ie:G.id-J.id}if(typeof performance=="object"&&typeof performance.now=="function"){var l=performance;s.unstable_now=function(){return l.now()}}else{var d=Date,m=d.now();s.unstable_now=function(){return d.now()-m}}var g=[],_=[],M=1,u=null,f=3,p=!1,y=!1,E=!1,S=typeof setTimeout=="function"?setTimeout:null,v=typeof clearTimeout=="function"?clearTimeout:null,A=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function P(G){for(var J=t(_);J!==null;){if(J.callback===null)r(_);else if(J.startTime<=G)r(_),J.sortIndex=J.expirationTime,e(g,J);else break;J=t(_)}}function L(G){if(E=!1,P(G),!y)if(t(g)!==null)y=!0,Z(z);else{var J=t(_);J!==null&&q(L,J.startTime-G)}}function z(G,J){y=!1,E&&(E=!1,v(R),R=-1),p=!0;var ie=f;try{for(P(J),u=t(g);u!==null&&(!(u.expirationTime>J)||G&&!O());){var U=u.callback;if(typeof U=="function"){u.callback=null,f=u.priorityLevel;var K=U(u.expirationTime<=J);J=s.unstable_now(),typeof K=="function"?u.callback=K:u===t(g)&&r(g),P(J)}else r(g);u=t(g)}if(u!==null)var Le=!0;else{var De=t(_);De!==null&&q(L,De.startTime-J),Le=!1}return Le}finally{u=null,f=ie,p=!1}}var D=!1,F=null,R=-1,I=5,W=-1;function O(){return!(s.unstable_now()-W<I)}function j(){if(F!==null){var G=s.unstable_now();W=G;var J=!0;try{J=F(!0,G)}finally{J?re():(D=!1,F=null)}}else D=!1}var re;if(typeof A=="function")re=function(){A(j)};else if(typeof MessageChannel<"u"){var ae=new MessageChannel,X=ae.port2;ae.port1.onmessage=j,re=function(){X.postMessage(null)}}else re=function(){S(j,0)};function Z(G){F=G,D||(D=!0,re())}function q(G,J){R=S(function(){G(s.unstable_now())},J)}s.unstable_IdlePriority=5,s.unstable_ImmediatePriority=1,s.unstable_LowPriority=4,s.unstable_NormalPriority=3,s.unstable_Profiling=null,s.unstable_UserBlockingPriority=2,s.unstable_cancelCallback=function(G){G.callback=null},s.unstable_continueExecution=function(){y||p||(y=!0,Z(z))},s.unstable_forceFrameRate=function(G){0>G||125<G?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):I=0<G?Math.floor(1e3/G):5},s.unstable_getCurrentPriorityLevel=function(){return f},s.unstable_getFirstCallbackNode=function(){return t(g)},s.unstable_next=function(G){switch(f){case 1:case 2:case 3:var J=3;break;default:J=f}var ie=f;f=J;try{return G()}finally{f=ie}},s.unstable_pauseExecution=function(){},s.unstable_requestPaint=function(){},s.unstable_runWithPriority=function(G,J){switch(G){case 1:case 2:case 3:case 4:case 5:break;default:G=3}var ie=f;f=G;try{return J()}finally{f=ie}},s.unstable_scheduleCallback=function(G,J,ie){var U=s.unstable_now();switch(typeof ie=="object"&&ie!==null?(ie=ie.delay,ie=typeof ie=="number"&&0<ie?U+ie:U):ie=U,G){case 1:var K=-1;break;case 2:K=250;break;case 5:K=1073741823;break;case 4:K=1e4;break;default:K=5e3}return K=ie+K,G={id:M++,callback:J,priorityLevel:G,startTime:ie,expirationTime:K,sortIndex:-1},ie>U?(G.sortIndex=ie,e(_,G),t(g)===null&&G===t(_)&&(E?(v(R),R=-1):E=!0,q(L,ie-U))):(G.sortIndex=K,e(g,G),y||p||(y=!0,Z(z))),G},s.unstable_shouldYield=O,s.unstable_wrapCallback=function(G){var J=f;return function(){var ie=f;f=J;try{return G.apply(this,arguments)}finally{f=ie}}}})(zc)),zc}var Jp;function C0(){return Jp||(Jp=1,kc.exports=R0()),kc.exports}/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var em;function b0(){if(em)return Bn;em=1;var s=hd(),e=C0();function t(n){for(var i="https://reactjs.org/docs/error-decoder.html?invariant="+n,o=1;o<arguments.length;o++)i+="&args[]="+encodeURIComponent(arguments[o]);return"Minified React error #"+n+"; visit "+i+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var r=new Set,a={};function l(n,i){d(n,i),d(n+"Capture",i)}function d(n,i){for(a[n]=i,n=0;n<i.length;n++)r.add(i[n])}var m=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),g=Object.prototype.hasOwnProperty,_=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,M={},u={};function f(n){return g.call(u,n)?!0:g.call(M,n)?!1:_.test(n)?u[n]=!0:(M[n]=!0,!1)}function p(n,i,o,c){if(o!==null&&o.type===0)return!1;switch(typeof i){case"function":case"symbol":return!0;case"boolean":return c?!1:o!==null?!o.acceptsBooleans:(n=n.toLowerCase().slice(0,5),n!=="data-"&&n!=="aria-");default:return!1}}function y(n,i,o,c){if(i===null||typeof i>"u"||p(n,i,o,c))return!0;if(c)return!1;if(o!==null)switch(o.type){case 3:return!i;case 4:return i===!1;case 5:return isNaN(i);case 6:return isNaN(i)||1>i}return!1}function E(n,i,o,c,h,x,w){this.acceptsBooleans=i===2||i===3||i===4,this.attributeName=c,this.attributeNamespace=h,this.mustUseProperty=o,this.propertyName=n,this.type=i,this.sanitizeURL=x,this.removeEmptyString=w}var S={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(n){S[n]=new E(n,0,!1,n,null,!1,!1)}),[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(n){var i=n[0];S[i]=new E(i,1,!1,n[1],null,!1,!1)}),["contentEditable","draggable","spellCheck","value"].forEach(function(n){S[n]=new E(n,2,!1,n.toLowerCase(),null,!1,!1)}),["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(n){S[n]=new E(n,2,!1,n,null,!1,!1)}),"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(n){S[n]=new E(n,3,!1,n.toLowerCase(),null,!1,!1)}),["checked","multiple","muted","selected"].forEach(function(n){S[n]=new E(n,3,!0,n,null,!1,!1)}),["capture","download"].forEach(function(n){S[n]=new E(n,4,!1,n,null,!1,!1)}),["cols","rows","size","span"].forEach(function(n){S[n]=new E(n,6,!1,n,null,!1,!1)}),["rowSpan","start"].forEach(function(n){S[n]=new E(n,5,!1,n.toLowerCase(),null,!1,!1)});var v=/[\-:]([a-z])/g;function A(n){return n[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(n){var i=n.replace(v,A);S[i]=new E(i,1,!1,n,null,!1,!1)}),"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(n){var i=n.replace(v,A);S[i]=new E(i,1,!1,n,"http://www.w3.org/1999/xlink",!1,!1)}),["xml:base","xml:lang","xml:space"].forEach(function(n){var i=n.replace(v,A);S[i]=new E(i,1,!1,n,"http://www.w3.org/XML/1998/namespace",!1,!1)}),["tabIndex","crossOrigin"].forEach(function(n){S[n]=new E(n,1,!1,n.toLowerCase(),null,!1,!1)}),S.xlinkHref=new E("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1),["src","href","action","formAction"].forEach(function(n){S[n]=new E(n,1,!1,n.toLowerCase(),null,!0,!0)});function P(n,i,o,c){var h=S.hasOwnProperty(i)?S[i]:null;(h!==null?h.type!==0:c||!(2<i.length)||i[0]!=="o"&&i[0]!=="O"||i[1]!=="n"&&i[1]!=="N")&&(y(i,o,h,c)&&(o=null),c||h===null?f(i)&&(o===null?n.removeAttribute(i):n.setAttribute(i,""+o)):h.mustUseProperty?n[h.propertyName]=o===null?h.type===3?!1:"":o:(i=h.attributeName,c=h.attributeNamespace,o===null?n.removeAttribute(i):(h=h.type,o=h===3||h===4&&o===!0?"":""+o,c?n.setAttributeNS(c,i,o):n.setAttribute(i,o))))}var L=s.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,z=Symbol.for("react.element"),D=Symbol.for("react.portal"),F=Symbol.for("react.fragment"),R=Symbol.for("react.strict_mode"),I=Symbol.for("react.profiler"),W=Symbol.for("react.provider"),O=Symbol.for("react.context"),j=Symbol.for("react.forward_ref"),re=Symbol.for("react.suspense"),ae=Symbol.for("react.suspense_list"),X=Symbol.for("react.memo"),Z=Symbol.for("react.lazy"),q=Symbol.for("react.offscreen"),G=Symbol.iterator;function J(n){return n===null||typeof n!="object"?null:(n=G&&n[G]||n["@@iterator"],typeof n=="function"?n:null)}var ie=Object.assign,U;function K(n){if(U===void 0)try{throw Error()}catch(o){var i=o.stack.trim().match(/\n( *(at )?)/);U=i&&i[1]||""}return`
`+U+n}var Le=!1;function De(n,i){if(!n||Le)return"";Le=!0;var o=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(i)if(i=function(){throw Error()},Object.defineProperty(i.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(i,[])}catch(ue){var c=ue}Reflect.construct(n,[],i)}else{try{i.call()}catch(ue){c=ue}n.call(i.prototype)}else{try{throw Error()}catch(ue){c=ue}n()}}catch(ue){if(ue&&c&&typeof ue.stack=="string"){for(var h=ue.stack.split(`
`),x=c.stack.split(`
`),w=h.length-1,N=x.length-1;1<=w&&0<=N&&h[w]!==x[N];)N--;for(;1<=w&&0<=N;w--,N--)if(h[w]!==x[N]){if(w!==1||N!==1)do if(w--,N--,0>N||h[w]!==x[N]){var B=`
`+h[w].replace(" at new "," at ");return n.displayName&&B.includes("<anonymous>")&&(B=B.replace("<anonymous>",n.displayName)),B}while(1<=w&&0<=N);break}}}finally{Le=!1,Error.prepareStackTrace=o}return(n=n?n.displayName||n.name:"")?K(n):""}function we(n){switch(n.tag){case 5:return K(n.type);case 16:return K("Lazy");case 13:return K("Suspense");case 19:return K("SuspenseList");case 0:case 2:case 15:return n=De(n.type,!1),n;case 11:return n=De(n.type.render,!1),n;case 1:return n=De(n.type,!0),n;default:return""}}function se(n){if(n==null)return null;if(typeof n=="function")return n.displayName||n.name||null;if(typeof n=="string")return n;switch(n){case F:return"Fragment";case D:return"Portal";case I:return"Profiler";case R:return"StrictMode";case re:return"Suspense";case ae:return"SuspenseList"}if(typeof n=="object")switch(n.$$typeof){case O:return(n.displayName||"Context")+".Consumer";case W:return(n._context.displayName||"Context")+".Provider";case j:var i=n.render;return n=n.displayName,n||(n=i.displayName||i.name||"",n=n!==""?"ForwardRef("+n+")":"ForwardRef"),n;case X:return i=n.displayName||null,i!==null?i:se(n.type)||"Memo";case Z:i=n._payload,n=n._init;try{return se(n(i))}catch{}}return null}function _e(n){var i=n.type;switch(n.tag){case 24:return"Cache";case 9:return(i.displayName||"Context")+".Consumer";case 10:return(i._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return n=i.render,n=n.displayName||n.name||"",i.displayName||(n!==""?"ForwardRef("+n+")":"ForwardRef");case 7:return"Fragment";case 5:return i;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return se(i);case 8:return i===R?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof i=="function")return i.displayName||i.name||null;if(typeof i=="string")return i}return null}function de(n){switch(typeof n){case"boolean":case"number":case"string":case"undefined":return n;case"object":return n;default:return""}}function Ie(n){var i=n.type;return(n=n.nodeName)&&n.toLowerCase()==="input"&&(i==="checkbox"||i==="radio")}function je(n){var i=Ie(n)?"checked":"value",o=Object.getOwnPropertyDescriptor(n.constructor.prototype,i),c=""+n[i];if(!n.hasOwnProperty(i)&&typeof o<"u"&&typeof o.get=="function"&&typeof o.set=="function"){var h=o.get,x=o.set;return Object.defineProperty(n,i,{configurable:!0,get:function(){return h.call(this)},set:function(w){c=""+w,x.call(this,w)}}),Object.defineProperty(n,i,{enumerable:o.enumerable}),{getValue:function(){return c},setValue:function(w){c=""+w},stopTracking:function(){n._valueTracker=null,delete n[i]}}}}function $e(n){n._valueTracker||(n._valueTracker=je(n))}function Ut(n){if(!n)return!1;var i=n._valueTracker;if(!i)return!0;var o=i.getValue(),c="";return n&&(c=Ie(n)?n.checked?"true":"false":n.value),n=c,n!==o?(i.setValue(n),!0):!1}function ct(n){if(n=n||(typeof document<"u"?document:void 0),typeof n>"u")return null;try{return n.activeElement||n.body}catch{return n.body}}function Et(n,i){var o=i.checked;return ie({},i,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:o??n._wrapperState.initialChecked})}function Dt(n,i){var o=i.defaultValue==null?"":i.defaultValue,c=i.checked!=null?i.checked:i.defaultChecked;o=de(i.value!=null?i.value:o),n._wrapperState={initialChecked:c,initialValue:o,controlled:i.type==="checkbox"||i.type==="radio"?i.checked!=null:i.value!=null}}function ft(n,i){i=i.checked,i!=null&&P(n,"checked",i,!1)}function Yt(n,i){ft(n,i);var o=de(i.value),c=i.type;if(o!=null)c==="number"?(o===0&&n.value===""||n.value!=o)&&(n.value=""+o):n.value!==""+o&&(n.value=""+o);else if(c==="submit"||c==="reset"){n.removeAttribute("value");return}i.hasOwnProperty("value")?hn(n,i.type,o):i.hasOwnProperty("defaultValue")&&hn(n,i.type,de(i.defaultValue)),i.checked==null&&i.defaultChecked!=null&&(n.defaultChecked=!!i.defaultChecked)}function Ft(n,i,o){if(i.hasOwnProperty("value")||i.hasOwnProperty("defaultValue")){var c=i.type;if(!(c!=="submit"&&c!=="reset"||i.value!==void 0&&i.value!==null))return;i=""+n._wrapperState.initialValue,o||i===n.value||(n.value=i),n.defaultValue=i}o=n.name,o!==""&&(n.name=""),n.defaultChecked=!!n._wrapperState.initialChecked,o!==""&&(n.name=o)}function hn(n,i,o){(i!=="number"||ct(n.ownerDocument)!==n)&&(o==null?n.defaultValue=""+n._wrapperState.initialValue:n.defaultValue!==""+o&&(n.defaultValue=""+o))}var H=Array.isArray;function Ot(n,i,o,c){if(n=n.options,i){i={};for(var h=0;h<o.length;h++)i["$"+o[h]]=!0;for(o=0;o<n.length;o++)h=i.hasOwnProperty("$"+n[o].value),n[o].selected!==h&&(n[o].selected=h),h&&c&&(n[o].defaultSelected=!0)}else{for(o=""+de(o),i=null,h=0;h<n.length;h++){if(n[h].value===o){n[h].selected=!0,c&&(n[h].defaultSelected=!0);return}i!==null||n[h].disabled||(i=n[h])}i!==null&&(i.selected=!0)}}function dt(n,i){if(i.dangerouslySetInnerHTML!=null)throw Error(t(91));return ie({},i,{value:void 0,defaultValue:void 0,children:""+n._wrapperState.initialValue})}function Ct(n,i){var o=i.value;if(o==null){if(o=i.children,i=i.defaultValue,o!=null){if(i!=null)throw Error(t(92));if(H(o)){if(1<o.length)throw Error(t(93));o=o[0]}i=o}i==null&&(i=""),o=i}n._wrapperState={initialValue:de(o)}}function Ne(n,i){var o=de(i.value),c=de(i.defaultValue);o!=null&&(o=""+o,o!==n.value&&(n.value=o),i.defaultValue==null&&n.defaultValue!==o&&(n.defaultValue=o)),c!=null&&(n.defaultValue=""+c)}function zt(n){var i=n.textContent;i===n._wrapperState.initialValue&&i!==""&&i!==null&&(n.value=i)}function b(n){switch(n){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function T(n,i){return n==null||n==="http://www.w3.org/1999/xhtml"?b(i):n==="http://www.w3.org/2000/svg"&&i==="foreignObject"?"http://www.w3.org/1999/xhtml":n}var $,he=(function(n){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(i,o,c,h){MSApp.execUnsafeLocalFunction(function(){return n(i,o,c,h)})}:n})(function(n,i){if(n.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in n)n.innerHTML=i;else{for($=$||document.createElement("div"),$.innerHTML="<svg>"+i.valueOf().toString()+"</svg>",i=$.firstChild;n.firstChild;)n.removeChild(n.firstChild);for(;i.firstChild;)n.appendChild(i.firstChild)}});function me(n,i){if(i){var o=n.firstChild;if(o&&o===n.lastChild&&o.nodeType===3){o.nodeValue=i;return}}n.textContent=i}var ye={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Pe=["Webkit","ms","Moz","O"];Object.keys(ye).forEach(function(n){Pe.forEach(function(i){i=i+n.charAt(0).toUpperCase()+n.substring(1),ye[i]=ye[n]})});function ce(n,i,o){return i==null||typeof i=="boolean"||i===""?"":o||typeof i!="number"||i===0||ye.hasOwnProperty(n)&&ye[n]?(""+i).trim():i+"px"}function pe(n,i){n=n.style;for(var o in i)if(i.hasOwnProperty(o)){var c=o.indexOf("--")===0,h=ce(o,i[o],c);o==="float"&&(o="cssFloat"),c?n.setProperty(o,h):n[o]=h}}var Fe=ie({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Be(n,i){if(i){if(Fe[n]&&(i.children!=null||i.dangerouslySetInnerHTML!=null))throw Error(t(137,n));if(i.dangerouslySetInnerHTML!=null){if(i.children!=null)throw Error(t(60));if(typeof i.dangerouslySetInnerHTML!="object"||!("__html"in i.dangerouslySetInnerHTML))throw Error(t(61))}if(i.style!=null&&typeof i.style!="object")throw Error(t(62))}}function Ae(n,i){if(n.indexOf("-")===-1)return typeof i.is=="string";switch(n){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Me=null;function et(n){return n=n.target||n.srcElement||window,n.correspondingUseElement&&(n=n.correspondingUseElement),n.nodeType===3?n.parentNode:n}var rt=null,pt=null,k=null;function Te(n){if(n=Ro(n)){if(typeof rt!="function")throw Error(t(280));var i=n.stateNode;i&&(i=Aa(i),rt(n.stateNode,n.type,i))}}function fe(n){pt?k?k.push(n):k=[n]:pt=n}function Oe(){if(pt){var n=pt,i=k;if(k=pt=null,Te(n),i)for(n=0;n<i.length;n++)Te(i[n])}}function Ce(n,i){return n(i)}function ge(){}var We=!1;function st(n,i,o){if(We)return n(i,o);We=!0;try{return Ce(n,i,o)}finally{We=!1,(pt!==null||k!==null)&&(ge(),Oe())}}function Nt(n,i){var o=n.stateNode;if(o===null)return null;var c=Aa(o);if(c===null)return null;o=c[i];e:switch(i){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(c=!c.disabled)||(n=n.type,c=!(n==="button"||n==="input"||n==="select"||n==="textarea")),n=!c;break e;default:n=!1}if(n)return null;if(o&&typeof o!="function")throw Error(t(231,i,typeof o));return o}var Tt=!1;if(m)try{var wn={};Object.defineProperty(wn,"passive",{get:function(){Tt=!0}}),window.addEventListener("test",wn,wn),window.removeEventListener("test",wn,wn)}catch{Tt=!1}function qn(n,i,o,c,h,x,w,N,B){var ue=Array.prototype.slice.call(arguments,3);try{i.apply(o,ue)}catch(xe){this.onError(xe)}}var Ui=!1,us=null,Ir=!1,cs=null,Fi={onError:function(n){Ui=!0,us=n}};function so(n,i,o,c,h,x,w,N,B){Ui=!1,us=null,qn.apply(Fi,arguments)}function ua(n,i,o,c,h,x,w,N,B){if(so.apply(this,arguments),Ui){if(Ui){var ue=us;Ui=!1,us=null}else throw Error(t(198));Ir||(Ir=!0,cs=ue)}}function Si(n){var i=n,o=n;if(n.alternate)for(;i.return;)i=i.return;else{n=i;do i=n,(i.flags&4098)!==0&&(o=i.return),n=i.return;while(n)}return i.tag===3?o:null}function Nr(n){if(n.tag===13){var i=n.memoizedState;if(i===null&&(n=n.alternate,n!==null&&(i=n.memoizedState)),i!==null)return i.dehydrated}return null}function oo(n){if(Si(n)!==n)throw Error(t(188))}function fs(n){var i=n.alternate;if(!i){if(i=Si(n),i===null)throw Error(t(188));return i!==n?null:n}for(var o=n,c=i;;){var h=o.return;if(h===null)break;var x=h.alternate;if(x===null){if(c=h.return,c!==null){o=c;continue}break}if(h.child===x.child){for(x=h.child;x;){if(x===o)return oo(h),n;if(x===c)return oo(h),i;x=x.sibling}throw Error(t(188))}if(o.return!==c.return)o=h,c=x;else{for(var w=!1,N=h.child;N;){if(N===o){w=!0,o=h,c=x;break}if(N===c){w=!0,c=h,o=x;break}N=N.sibling}if(!w){for(N=x.child;N;){if(N===o){w=!0,o=x,c=h;break}if(N===c){w=!0,c=x,o=h;break}N=N.sibling}if(!w)throw Error(t(189))}}if(o.alternate!==c)throw Error(t(190))}if(o.tag!==3)throw Error(t(188));return o.stateNode.current===o?n:i}function ao(n){return n=fs(n),n!==null?lo(n):null}function lo(n){if(n.tag===5||n.tag===6)return n;for(n=n.child;n!==null;){var i=lo(n);if(i!==null)return i;n=n.sibling}return null}var ca=e.unstable_scheduleCallback,fa=e.unstable_cancelCallback,su=e.unstable_shouldYield,ou=e.unstable_requestPaint,qt=e.unstable_now,au=e.unstable_getCurrentPriorityLevel,uo=e.unstable_ImmediatePriority,C=e.unstable_UserBlockingPriority,Y=e.unstable_NormalPriority,le=e.unstable_LowPriority,te=e.unstable_IdlePriority,ee=null,be=null;function He(n){if(be&&typeof be.onCommitFiberRoot=="function")try{be.onCommitFiberRoot(ee,n,void 0,(n.current.flags&128)===128)}catch{}}var Re=Math.clz32?Math.clz32:ot,Xe=Math.log,Ze=Math.LN2;function ot(n){return n>>>=0,n===0?32:31-(Xe(n)/Ze|0)|0}var at=64,qe=4194304;function St(n){switch(n&-n){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return n&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return n&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return n}}function Bt(n,i){var o=n.pendingLanes;if(o===0)return 0;var c=0,h=n.suspendedLanes,x=n.pingedLanes,w=o&268435455;if(w!==0){var N=w&~h;N!==0?c=St(N):(x&=w,x!==0&&(c=St(x)))}else w=o&~h,w!==0?c=St(w):x!==0&&(c=St(x));if(c===0)return 0;if(i!==0&&i!==c&&(i&h)===0&&(h=c&-c,x=i&-i,h>=x||h===16&&(x&4194240)!==0))return i;if((c&4)!==0&&(c|=o&16),i=n.entangledLanes,i!==0)for(n=n.entanglements,i&=c;0<i;)o=31-Re(i),h=1<<o,c|=n[o],i&=~h;return c}function Wt(n,i){switch(n){case 1:case 2:case 4:return i+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return i+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function bt(n,i){for(var o=n.suspendedLanes,c=n.pingedLanes,h=n.expirationTimes,x=n.pendingLanes;0<x;){var w=31-Re(x),N=1<<w,B=h[w];B===-1?((N&o)===0||(N&c)!==0)&&(h[w]=Wt(N,i)):B<=i&&(n.expiredLanes|=N),x&=~N}}function en(n){return n=n.pendingLanes&-1073741825,n!==0?n:n&1073741824?1073741824:0}function ke(){var n=at;return at<<=1,(at&4194240)===0&&(at=64),n}function pn(n){for(var i=[],o=0;31>o;o++)i.push(n);return i}function mt(n,i,o){n.pendingLanes|=i,i!==536870912&&(n.suspendedLanes=0,n.pingedLanes=0),n=n.eventTimes,i=31-Re(i),n[i]=o}function Ln(n,i){var o=n.pendingLanes&~i;n.pendingLanes=i,n.suspendedLanes=0,n.pingedLanes=0,n.expiredLanes&=i,n.mutableReadLanes&=i,n.entangledLanes&=i,i=n.entanglements;var c=n.eventTimes;for(n=n.expirationTimes;0<o;){var h=31-Re(o),x=1<<h;i[h]=0,c[h]=-1,n[h]=-1,o&=~x}}function Dn(n,i){var o=n.entangledLanes|=i;for(n=n.entanglements;o;){var c=31-Re(o),h=1<<c;h&i|n[c]&i&&(n[c]|=i),o&=~h}}var _t=0;function Oi(n){return n&=-n,1<n?4<n?(n&268435455)!==0?16:536870912:4:1}var Rt,Ht,si,Pt,oi,yi=!1,Ur=[],sr=null,or=null,ar=null,co=new Map,fo=new Map,lr=[],Y_="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function Ld(n,i){switch(n){case"focusin":case"focusout":sr=null;break;case"dragenter":case"dragleave":or=null;break;case"mouseover":case"mouseout":ar=null;break;case"pointerover":case"pointerout":co.delete(i.pointerId);break;case"gotpointercapture":case"lostpointercapture":fo.delete(i.pointerId)}}function ho(n,i,o,c,h,x){return n===null||n.nativeEvent!==x?(n={blockedOn:i,domEventName:o,eventSystemFlags:c,nativeEvent:x,targetContainers:[h]},i!==null&&(i=Ro(i),i!==null&&Ht(i)),n):(n.eventSystemFlags|=c,i=n.targetContainers,h!==null&&i.indexOf(h)===-1&&i.push(h),n)}function q_(n,i,o,c,h){switch(i){case"focusin":return sr=ho(sr,n,i,o,c,h),!0;case"dragenter":return or=ho(or,n,i,o,c,h),!0;case"mouseover":return ar=ho(ar,n,i,o,c,h),!0;case"pointerover":var x=h.pointerId;return co.set(x,ho(co.get(x)||null,n,i,o,c,h)),!0;case"gotpointercapture":return x=h.pointerId,fo.set(x,ho(fo.get(x)||null,n,i,o,c,h)),!0}return!1}function Dd(n){var i=Fr(n.target);if(i!==null){var o=Si(i);if(o!==null){if(i=o.tag,i===13){if(i=Nr(o),i!==null){n.blockedOn=i,oi(n.priority,function(){si(o)});return}}else if(i===3&&o.stateNode.current.memoizedState.isDehydrated){n.blockedOn=o.tag===3?o.stateNode.containerInfo:null;return}}}n.blockedOn=null}function da(n){if(n.blockedOn!==null)return!1;for(var i=n.targetContainers;0<i.length;){var o=uu(n.domEventName,n.eventSystemFlags,i[0],n.nativeEvent);if(o===null){o=n.nativeEvent;var c=new o.constructor(o.type,o);Me=c,o.target.dispatchEvent(c),Me=null}else return i=Ro(o),i!==null&&Ht(i),n.blockedOn=o,!1;i.shift()}return!0}function Id(n,i,o){da(n)&&o.delete(i)}function j_(){yi=!1,sr!==null&&da(sr)&&(sr=null),or!==null&&da(or)&&(or=null),ar!==null&&da(ar)&&(ar=null),co.forEach(Id),fo.forEach(Id)}function po(n,i){n.blockedOn===i&&(n.blockedOn=null,yi||(yi=!0,e.unstable_scheduleCallback(e.unstable_NormalPriority,j_)))}function mo(n){function i(h){return po(h,n)}if(0<Ur.length){po(Ur[0],n);for(var o=1;o<Ur.length;o++){var c=Ur[o];c.blockedOn===n&&(c.blockedOn=null)}}for(sr!==null&&po(sr,n),or!==null&&po(or,n),ar!==null&&po(ar,n),co.forEach(i),fo.forEach(i),o=0;o<lr.length;o++)c=lr[o],c.blockedOn===n&&(c.blockedOn=null);for(;0<lr.length&&(o=lr[0],o.blockedOn===null);)Dd(o),o.blockedOn===null&&lr.shift()}var ds=L.ReactCurrentBatchConfig,ha=!0;function K_(n,i,o,c){var h=_t,x=ds.transition;ds.transition=null;try{_t=1,lu(n,i,o,c)}finally{_t=h,ds.transition=x}}function $_(n,i,o,c){var h=_t,x=ds.transition;ds.transition=null;try{_t=4,lu(n,i,o,c)}finally{_t=h,ds.transition=x}}function lu(n,i,o,c){if(ha){var h=uu(n,i,o,c);if(h===null)Au(n,i,c,pa,o),Ld(n,c);else if(q_(h,n,i,o,c))c.stopPropagation();else if(Ld(n,c),i&4&&-1<Y_.indexOf(n)){for(;h!==null;){var x=Ro(h);if(x!==null&&Rt(x),x=uu(n,i,o,c),x===null&&Au(n,i,c,pa,o),x===h)break;h=x}h!==null&&c.stopPropagation()}else Au(n,i,c,null,o)}}var pa=null;function uu(n,i,o,c){if(pa=null,n=et(c),n=Fr(n),n!==null)if(i=Si(n),i===null)n=null;else if(o=i.tag,o===13){if(n=Nr(i),n!==null)return n;n=null}else if(o===3){if(i.stateNode.current.memoizedState.isDehydrated)return i.tag===3?i.stateNode.containerInfo:null;n=null}else i!==n&&(n=null);return pa=n,null}function Nd(n){switch(n){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(au()){case uo:return 1;case C:return 4;case Y:case le:return 16;case te:return 536870912;default:return 16}default:return 16}}var ur=null,cu=null,ma=null;function Ud(){if(ma)return ma;var n,i=cu,o=i.length,c,h="value"in ur?ur.value:ur.textContent,x=h.length;for(n=0;n<o&&i[n]===h[n];n++);var w=o-n;for(c=1;c<=w&&i[o-c]===h[x-c];c++);return ma=h.slice(n,1<c?1-c:void 0)}function _a(n){var i=n.keyCode;return"charCode"in n?(n=n.charCode,n===0&&i===13&&(n=13)):n=i,n===10&&(n=13),32<=n||n===13?n:0}function ga(){return!0}function Fd(){return!1}function Hn(n){function i(o,c,h,x,w){this._reactName=o,this._targetInst=h,this.type=c,this.nativeEvent=x,this.target=w,this.currentTarget=null;for(var N in n)n.hasOwnProperty(N)&&(o=n[N],this[N]=o?o(x):x[N]);return this.isDefaultPrevented=(x.defaultPrevented!=null?x.defaultPrevented:x.returnValue===!1)?ga:Fd,this.isPropagationStopped=Fd,this}return ie(i.prototype,{preventDefault:function(){this.defaultPrevented=!0;var o=this.nativeEvent;o&&(o.preventDefault?o.preventDefault():typeof o.returnValue!="unknown"&&(o.returnValue=!1),this.isDefaultPrevented=ga)},stopPropagation:function(){var o=this.nativeEvent;o&&(o.stopPropagation?o.stopPropagation():typeof o.cancelBubble!="unknown"&&(o.cancelBubble=!0),this.isPropagationStopped=ga)},persist:function(){},isPersistent:ga}),i}var hs={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(n){return n.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},fu=Hn(hs),_o=ie({},hs,{view:0,detail:0}),Z_=Hn(_o),du,hu,go,va=ie({},_o,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:mu,button:0,buttons:0,relatedTarget:function(n){return n.relatedTarget===void 0?n.fromElement===n.srcElement?n.toElement:n.fromElement:n.relatedTarget},movementX:function(n){return"movementX"in n?n.movementX:(n!==go&&(go&&n.type==="mousemove"?(du=n.screenX-go.screenX,hu=n.screenY-go.screenY):hu=du=0,go=n),du)},movementY:function(n){return"movementY"in n?n.movementY:hu}}),Od=Hn(va),Q_=ie({},va,{dataTransfer:0}),J_=Hn(Q_),eg=ie({},_o,{relatedTarget:0}),pu=Hn(eg),tg=ie({},hs,{animationName:0,elapsedTime:0,pseudoElement:0}),ng=Hn(tg),ig=ie({},hs,{clipboardData:function(n){return"clipboardData"in n?n.clipboardData:window.clipboardData}}),rg=Hn(ig),sg=ie({},hs,{data:0}),Bd=Hn(sg),og={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},ag={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},lg={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function ug(n){var i=this.nativeEvent;return i.getModifierState?i.getModifierState(n):(n=lg[n])?!!i[n]:!1}function mu(){return ug}var cg=ie({},_o,{key:function(n){if(n.key){var i=og[n.key]||n.key;if(i!=="Unidentified")return i}return n.type==="keypress"?(n=_a(n),n===13?"Enter":String.fromCharCode(n)):n.type==="keydown"||n.type==="keyup"?ag[n.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:mu,charCode:function(n){return n.type==="keypress"?_a(n):0},keyCode:function(n){return n.type==="keydown"||n.type==="keyup"?n.keyCode:0},which:function(n){return n.type==="keypress"?_a(n):n.type==="keydown"||n.type==="keyup"?n.keyCode:0}}),fg=Hn(cg),dg=ie({},va,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),kd=Hn(dg),hg=ie({},_o,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:mu}),pg=Hn(hg),mg=ie({},hs,{propertyName:0,elapsedTime:0,pseudoElement:0}),_g=Hn(mg),gg=ie({},va,{deltaX:function(n){return"deltaX"in n?n.deltaX:"wheelDeltaX"in n?-n.wheelDeltaX:0},deltaY:function(n){return"deltaY"in n?n.deltaY:"wheelDeltaY"in n?-n.wheelDeltaY:"wheelDelta"in n?-n.wheelDelta:0},deltaZ:0,deltaMode:0}),vg=Hn(gg),xg=[9,13,27,32],_u=m&&"CompositionEvent"in window,vo=null;m&&"documentMode"in document&&(vo=document.documentMode);var Sg=m&&"TextEvent"in window&&!vo,zd=m&&(!_u||vo&&8<vo&&11>=vo),Hd=" ",Vd=!1;function Gd(n,i){switch(n){case"keyup":return xg.indexOf(i.keyCode)!==-1;case"keydown":return i.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Wd(n){return n=n.detail,typeof n=="object"&&"data"in n?n.data:null}var ps=!1;function yg(n,i){switch(n){case"compositionend":return Wd(i);case"keypress":return i.which!==32?null:(Vd=!0,Hd);case"textInput":return n=i.data,n===Hd&&Vd?null:n;default:return null}}function Mg(n,i){if(ps)return n==="compositionend"||!_u&&Gd(n,i)?(n=Ud(),ma=cu=ur=null,ps=!1,n):null;switch(n){case"paste":return null;case"keypress":if(!(i.ctrlKey||i.altKey||i.metaKey)||i.ctrlKey&&i.altKey){if(i.char&&1<i.char.length)return i.char;if(i.which)return String.fromCharCode(i.which)}return null;case"compositionend":return zd&&i.locale!=="ko"?null:i.data;default:return null}}var Eg={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Xd(n){var i=n&&n.nodeName&&n.nodeName.toLowerCase();return i==="input"?!!Eg[n.type]:i==="textarea"}function Yd(n,i,o,c){fe(c),i=Ea(i,"onChange"),0<i.length&&(o=new fu("onChange","change",null,o,c),n.push({event:o,listeners:i}))}var xo=null,So=null;function Tg(n){ch(n,0)}function xa(n){var i=xs(n);if(Ut(i))return n}function wg(n,i){if(n==="change")return i}var qd=!1;if(m){var gu;if(m){var vu="oninput"in document;if(!vu){var jd=document.createElement("div");jd.setAttribute("oninput","return;"),vu=typeof jd.oninput=="function"}gu=vu}else gu=!1;qd=gu&&(!document.documentMode||9<document.documentMode)}function Kd(){xo&&(xo.detachEvent("onpropertychange",$d),So=xo=null)}function $d(n){if(n.propertyName==="value"&&xa(So)){var i=[];Yd(i,So,n,et(n)),st(Tg,i)}}function Ag(n,i,o){n==="focusin"?(Kd(),xo=i,So=o,xo.attachEvent("onpropertychange",$d)):n==="focusout"&&Kd()}function Rg(n){if(n==="selectionchange"||n==="keyup"||n==="keydown")return xa(So)}function Cg(n,i){if(n==="click")return xa(i)}function bg(n,i){if(n==="input"||n==="change")return xa(i)}function Pg(n,i){return n===i&&(n!==0||1/n===1/i)||n!==n&&i!==i}var ai=typeof Object.is=="function"?Object.is:Pg;function yo(n,i){if(ai(n,i))return!0;if(typeof n!="object"||n===null||typeof i!="object"||i===null)return!1;var o=Object.keys(n),c=Object.keys(i);if(o.length!==c.length)return!1;for(c=0;c<o.length;c++){var h=o[c];if(!g.call(i,h)||!ai(n[h],i[h]))return!1}return!0}function Zd(n){for(;n&&n.firstChild;)n=n.firstChild;return n}function Qd(n,i){var o=Zd(n);n=0;for(var c;o;){if(o.nodeType===3){if(c=n+o.textContent.length,n<=i&&c>=i)return{node:o,offset:i-n};n=c}e:{for(;o;){if(o.nextSibling){o=o.nextSibling;break e}o=o.parentNode}o=void 0}o=Zd(o)}}function Jd(n,i){return n&&i?n===i?!0:n&&n.nodeType===3?!1:i&&i.nodeType===3?Jd(n,i.parentNode):"contains"in n?n.contains(i):n.compareDocumentPosition?!!(n.compareDocumentPosition(i)&16):!1:!1}function eh(){for(var n=window,i=ct();i instanceof n.HTMLIFrameElement;){try{var o=typeof i.contentWindow.location.href=="string"}catch{o=!1}if(o)n=i.contentWindow;else break;i=ct(n.document)}return i}function xu(n){var i=n&&n.nodeName&&n.nodeName.toLowerCase();return i&&(i==="input"&&(n.type==="text"||n.type==="search"||n.type==="tel"||n.type==="url"||n.type==="password")||i==="textarea"||n.contentEditable==="true")}function Lg(n){var i=eh(),o=n.focusedElem,c=n.selectionRange;if(i!==o&&o&&o.ownerDocument&&Jd(o.ownerDocument.documentElement,o)){if(c!==null&&xu(o)){if(i=c.start,n=c.end,n===void 0&&(n=i),"selectionStart"in o)o.selectionStart=i,o.selectionEnd=Math.min(n,o.value.length);else if(n=(i=o.ownerDocument||document)&&i.defaultView||window,n.getSelection){n=n.getSelection();var h=o.textContent.length,x=Math.min(c.start,h);c=c.end===void 0?x:Math.min(c.end,h),!n.extend&&x>c&&(h=c,c=x,x=h),h=Qd(o,x);var w=Qd(o,c);h&&w&&(n.rangeCount!==1||n.anchorNode!==h.node||n.anchorOffset!==h.offset||n.focusNode!==w.node||n.focusOffset!==w.offset)&&(i=i.createRange(),i.setStart(h.node,h.offset),n.removeAllRanges(),x>c?(n.addRange(i),n.extend(w.node,w.offset)):(i.setEnd(w.node,w.offset),n.addRange(i)))}}for(i=[],n=o;n=n.parentNode;)n.nodeType===1&&i.push({element:n,left:n.scrollLeft,top:n.scrollTop});for(typeof o.focus=="function"&&o.focus(),o=0;o<i.length;o++)n=i[o],n.element.scrollLeft=n.left,n.element.scrollTop=n.top}}var Dg=m&&"documentMode"in document&&11>=document.documentMode,ms=null,Su=null,Mo=null,yu=!1;function th(n,i,o){var c=o.window===o?o.document:o.nodeType===9?o:o.ownerDocument;yu||ms==null||ms!==ct(c)||(c=ms,"selectionStart"in c&&xu(c)?c={start:c.selectionStart,end:c.selectionEnd}:(c=(c.ownerDocument&&c.ownerDocument.defaultView||window).getSelection(),c={anchorNode:c.anchorNode,anchorOffset:c.anchorOffset,focusNode:c.focusNode,focusOffset:c.focusOffset}),Mo&&yo(Mo,c)||(Mo=c,c=Ea(Su,"onSelect"),0<c.length&&(i=new fu("onSelect","select",null,i,o),n.push({event:i,listeners:c}),i.target=ms)))}function Sa(n,i){var o={};return o[n.toLowerCase()]=i.toLowerCase(),o["Webkit"+n]="webkit"+i,o["Moz"+n]="moz"+i,o}var _s={animationend:Sa("Animation","AnimationEnd"),animationiteration:Sa("Animation","AnimationIteration"),animationstart:Sa("Animation","AnimationStart"),transitionend:Sa("Transition","TransitionEnd")},Mu={},nh={};m&&(nh=document.createElement("div").style,"AnimationEvent"in window||(delete _s.animationend.animation,delete _s.animationiteration.animation,delete _s.animationstart.animation),"TransitionEvent"in window||delete _s.transitionend.transition);function ya(n){if(Mu[n])return Mu[n];if(!_s[n])return n;var i=_s[n],o;for(o in i)if(i.hasOwnProperty(o)&&o in nh)return Mu[n]=i[o];return n}var ih=ya("animationend"),rh=ya("animationiteration"),sh=ya("animationstart"),oh=ya("transitionend"),ah=new Map,lh="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function cr(n,i){ah.set(n,i),l(i,[n])}for(var Eu=0;Eu<lh.length;Eu++){var Tu=lh[Eu],Ig=Tu.toLowerCase(),Ng=Tu[0].toUpperCase()+Tu.slice(1);cr(Ig,"on"+Ng)}cr(ih,"onAnimationEnd"),cr(rh,"onAnimationIteration"),cr(sh,"onAnimationStart"),cr("dblclick","onDoubleClick"),cr("focusin","onFocus"),cr("focusout","onBlur"),cr(oh,"onTransitionEnd"),d("onMouseEnter",["mouseout","mouseover"]),d("onMouseLeave",["mouseout","mouseover"]),d("onPointerEnter",["pointerout","pointerover"]),d("onPointerLeave",["pointerout","pointerover"]),l("onChange","change click focusin focusout input keydown keyup selectionchange".split(" ")),l("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" ")),l("onBeforeInput",["compositionend","keypress","textInput","paste"]),l("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" ")),l("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" ")),l("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Eo="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),Ug=new Set("cancel close invalid load scroll toggle".split(" ").concat(Eo));function uh(n,i,o){var c=n.type||"unknown-event";n.currentTarget=o,ua(c,i,void 0,n),n.currentTarget=null}function ch(n,i){i=(i&4)!==0;for(var o=0;o<n.length;o++){var c=n[o],h=c.event;c=c.listeners;e:{var x=void 0;if(i)for(var w=c.length-1;0<=w;w--){var N=c[w],B=N.instance,ue=N.currentTarget;if(N=N.listener,B!==x&&h.isPropagationStopped())break e;uh(h,N,ue),x=B}else for(w=0;w<c.length;w++){if(N=c[w],B=N.instance,ue=N.currentTarget,N=N.listener,B!==x&&h.isPropagationStopped())break e;uh(h,N,ue),x=B}}}if(Ir)throw n=cs,Ir=!1,cs=null,n}function Vt(n,i){var o=i[Du];o===void 0&&(o=i[Du]=new Set);var c=n+"__bubble";o.has(c)||(fh(i,n,2,!1),o.add(c))}function wu(n,i,o){var c=0;i&&(c|=4),fh(o,n,c,i)}var Ma="_reactListening"+Math.random().toString(36).slice(2);function To(n){if(!n[Ma]){n[Ma]=!0,r.forEach(function(o){o!=="selectionchange"&&(Ug.has(o)||wu(o,!1,n),wu(o,!0,n))});var i=n.nodeType===9?n:n.ownerDocument;i===null||i[Ma]||(i[Ma]=!0,wu("selectionchange",!1,i))}}function fh(n,i,o,c){switch(Nd(i)){case 1:var h=K_;break;case 4:h=$_;break;default:h=lu}o=h.bind(null,i,o,n),h=void 0,!Tt||i!=="touchstart"&&i!=="touchmove"&&i!=="wheel"||(h=!0),c?h!==void 0?n.addEventListener(i,o,{capture:!0,passive:h}):n.addEventListener(i,o,!0):h!==void 0?n.addEventListener(i,o,{passive:h}):n.addEventListener(i,o,!1)}function Au(n,i,o,c,h){var x=c;if((i&1)===0&&(i&2)===0&&c!==null)e:for(;;){if(c===null)return;var w=c.tag;if(w===3||w===4){var N=c.stateNode.containerInfo;if(N===h||N.nodeType===8&&N.parentNode===h)break;if(w===4)for(w=c.return;w!==null;){var B=w.tag;if((B===3||B===4)&&(B=w.stateNode.containerInfo,B===h||B.nodeType===8&&B.parentNode===h))return;w=w.return}for(;N!==null;){if(w=Fr(N),w===null)return;if(B=w.tag,B===5||B===6){c=x=w;continue e}N=N.parentNode}}c=c.return}st(function(){var ue=x,xe=et(o),Se=[];e:{var ve=ah.get(n);if(ve!==void 0){var ze=fu,Ge=n;switch(n){case"keypress":if(_a(o)===0)break e;case"keydown":case"keyup":ze=fg;break;case"focusin":Ge="focus",ze=pu;break;case"focusout":Ge="blur",ze=pu;break;case"beforeblur":case"afterblur":ze=pu;break;case"click":if(o.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":ze=Od;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":ze=J_;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":ze=pg;break;case ih:case rh:case sh:ze=ng;break;case oh:ze=_g;break;case"scroll":ze=Z_;break;case"wheel":ze=vg;break;case"copy":case"cut":case"paste":ze=rg;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":ze=kd}var Ye=(i&4)!==0,Qt=!Ye&&n==="scroll",Q=Ye?ve!==null?ve+"Capture":null:ve;Ye=[];for(var V=ue,ne;V!==null;){ne=V;var Ee=ne.stateNode;if(ne.tag===5&&Ee!==null&&(ne=Ee,Q!==null&&(Ee=Nt(V,Q),Ee!=null&&Ye.push(wo(V,Ee,ne)))),Qt)break;V=V.return}0<Ye.length&&(ve=new ze(ve,Ge,null,o,xe),Se.push({event:ve,listeners:Ye}))}}if((i&7)===0){e:{if(ve=n==="mouseover"||n==="pointerover",ze=n==="mouseout"||n==="pointerout",ve&&o!==Me&&(Ge=o.relatedTarget||o.fromElement)&&(Fr(Ge)||Ge[Bi]))break e;if((ze||ve)&&(ve=xe.window===xe?xe:(ve=xe.ownerDocument)?ve.defaultView||ve.parentWindow:window,ze?(Ge=o.relatedTarget||o.toElement,ze=ue,Ge=Ge?Fr(Ge):null,Ge!==null&&(Qt=Si(Ge),Ge!==Qt||Ge.tag!==5&&Ge.tag!==6)&&(Ge=null)):(ze=null,Ge=ue),ze!==Ge)){if(Ye=Od,Ee="onMouseLeave",Q="onMouseEnter",V="mouse",(n==="pointerout"||n==="pointerover")&&(Ye=kd,Ee="onPointerLeave",Q="onPointerEnter",V="pointer"),Qt=ze==null?ve:xs(ze),ne=Ge==null?ve:xs(Ge),ve=new Ye(Ee,V+"leave",ze,o,xe),ve.target=Qt,ve.relatedTarget=ne,Ee=null,Fr(xe)===ue&&(Ye=new Ye(Q,V+"enter",Ge,o,xe),Ye.target=ne,Ye.relatedTarget=Qt,Ee=Ye),Qt=Ee,ze&&Ge)t:{for(Ye=ze,Q=Ge,V=0,ne=Ye;ne;ne=gs(ne))V++;for(ne=0,Ee=Q;Ee;Ee=gs(Ee))ne++;for(;0<V-ne;)Ye=gs(Ye),V--;for(;0<ne-V;)Q=gs(Q),ne--;for(;V--;){if(Ye===Q||Q!==null&&Ye===Q.alternate)break t;Ye=gs(Ye),Q=gs(Q)}Ye=null}else Ye=null;ze!==null&&dh(Se,ve,ze,Ye,!1),Ge!==null&&Qt!==null&&dh(Se,Qt,Ge,Ye,!0)}}e:{if(ve=ue?xs(ue):window,ze=ve.nodeName&&ve.nodeName.toLowerCase(),ze==="select"||ze==="input"&&ve.type==="file")var Ke=wg;else if(Xd(ve))if(qd)Ke=bg;else{Ke=Rg;var Qe=Ag}else(ze=ve.nodeName)&&ze.toLowerCase()==="input"&&(ve.type==="checkbox"||ve.type==="radio")&&(Ke=Cg);if(Ke&&(Ke=Ke(n,ue))){Yd(Se,Ke,o,xe);break e}Qe&&Qe(n,ve,ue),n==="focusout"&&(Qe=ve._wrapperState)&&Qe.controlled&&ve.type==="number"&&hn(ve,"number",ve.value)}switch(Qe=ue?xs(ue):window,n){case"focusin":(Xd(Qe)||Qe.contentEditable==="true")&&(ms=Qe,Su=ue,Mo=null);break;case"focusout":Mo=Su=ms=null;break;case"mousedown":yu=!0;break;case"contextmenu":case"mouseup":case"dragend":yu=!1,th(Se,o,xe);break;case"selectionchange":if(Dg)break;case"keydown":case"keyup":th(Se,o,xe)}var Je;if(_u)e:{switch(n){case"compositionstart":var nt="onCompositionStart";break e;case"compositionend":nt="onCompositionEnd";break e;case"compositionupdate":nt="onCompositionUpdate";break e}nt=void 0}else ps?Gd(n,o)&&(nt="onCompositionEnd"):n==="keydown"&&o.keyCode===229&&(nt="onCompositionStart");nt&&(zd&&o.locale!=="ko"&&(ps||nt!=="onCompositionStart"?nt==="onCompositionEnd"&&ps&&(Je=Ud()):(ur=xe,cu="value"in ur?ur.value:ur.textContent,ps=!0)),Qe=Ea(ue,nt),0<Qe.length&&(nt=new Bd(nt,n,null,o,xe),Se.push({event:nt,listeners:Qe}),Je?nt.data=Je:(Je=Wd(o),Je!==null&&(nt.data=Je)))),(Je=Sg?yg(n,o):Mg(n,o))&&(ue=Ea(ue,"onBeforeInput"),0<ue.length&&(xe=new Bd("onBeforeInput","beforeinput",null,o,xe),Se.push({event:xe,listeners:ue}),xe.data=Je))}ch(Se,i)})}function wo(n,i,o){return{instance:n,listener:i,currentTarget:o}}function Ea(n,i){for(var o=i+"Capture",c=[];n!==null;){var h=n,x=h.stateNode;h.tag===5&&x!==null&&(h=x,x=Nt(n,o),x!=null&&c.unshift(wo(n,x,h)),x=Nt(n,i),x!=null&&c.push(wo(n,x,h))),n=n.return}return c}function gs(n){if(n===null)return null;do n=n.return;while(n&&n.tag!==5);return n||null}function dh(n,i,o,c,h){for(var x=i._reactName,w=[];o!==null&&o!==c;){var N=o,B=N.alternate,ue=N.stateNode;if(B!==null&&B===c)break;N.tag===5&&ue!==null&&(N=ue,h?(B=Nt(o,x),B!=null&&w.unshift(wo(o,B,N))):h||(B=Nt(o,x),B!=null&&w.push(wo(o,B,N)))),o=o.return}w.length!==0&&n.push({event:i,listeners:w})}var Fg=/\r\n?/g,Og=/\u0000|\uFFFD/g;function hh(n){return(typeof n=="string"?n:""+n).replace(Fg,`
`).replace(Og,"")}function Ta(n,i,o){if(i=hh(i),hh(n)!==i&&o)throw Error(t(425))}function wa(){}var Ru=null,Cu=null;function bu(n,i){return n==="textarea"||n==="noscript"||typeof i.children=="string"||typeof i.children=="number"||typeof i.dangerouslySetInnerHTML=="object"&&i.dangerouslySetInnerHTML!==null&&i.dangerouslySetInnerHTML.__html!=null}var Pu=typeof setTimeout=="function"?setTimeout:void 0,Bg=typeof clearTimeout=="function"?clearTimeout:void 0,ph=typeof Promise=="function"?Promise:void 0,kg=typeof queueMicrotask=="function"?queueMicrotask:typeof ph<"u"?function(n){return ph.resolve(null).then(n).catch(zg)}:Pu;function zg(n){setTimeout(function(){throw n})}function Lu(n,i){var o=i,c=0;do{var h=o.nextSibling;if(n.removeChild(o),h&&h.nodeType===8)if(o=h.data,o==="/$"){if(c===0){n.removeChild(h),mo(i);return}c--}else o!=="$"&&o!=="$?"&&o!=="$!"||c++;o=h}while(o);mo(i)}function fr(n){for(;n!=null;n=n.nextSibling){var i=n.nodeType;if(i===1||i===3)break;if(i===8){if(i=n.data,i==="$"||i==="$!"||i==="$?")break;if(i==="/$")return null}}return n}function mh(n){n=n.previousSibling;for(var i=0;n;){if(n.nodeType===8){var o=n.data;if(o==="$"||o==="$!"||o==="$?"){if(i===0)return n;i--}else o==="/$"&&i++}n=n.previousSibling}return null}var vs=Math.random().toString(36).slice(2),Mi="__reactFiber$"+vs,Ao="__reactProps$"+vs,Bi="__reactContainer$"+vs,Du="__reactEvents$"+vs,Hg="__reactListeners$"+vs,Vg="__reactHandles$"+vs;function Fr(n){var i=n[Mi];if(i)return i;for(var o=n.parentNode;o;){if(i=o[Bi]||o[Mi]){if(o=i.alternate,i.child!==null||o!==null&&o.child!==null)for(n=mh(n);n!==null;){if(o=n[Mi])return o;n=mh(n)}return i}n=o,o=n.parentNode}return null}function Ro(n){return n=n[Mi]||n[Bi],!n||n.tag!==5&&n.tag!==6&&n.tag!==13&&n.tag!==3?null:n}function xs(n){if(n.tag===5||n.tag===6)return n.stateNode;throw Error(t(33))}function Aa(n){return n[Ao]||null}var Iu=[],Ss=-1;function dr(n){return{current:n}}function Gt(n){0>Ss||(n.current=Iu[Ss],Iu[Ss]=null,Ss--)}function kt(n,i){Ss++,Iu[Ss]=n.current,n.current=i}var hr={},vn=dr(hr),In=dr(!1),Or=hr;function ys(n,i){var o=n.type.contextTypes;if(!o)return hr;var c=n.stateNode;if(c&&c.__reactInternalMemoizedUnmaskedChildContext===i)return c.__reactInternalMemoizedMaskedChildContext;var h={},x;for(x in o)h[x]=i[x];return c&&(n=n.stateNode,n.__reactInternalMemoizedUnmaskedChildContext=i,n.__reactInternalMemoizedMaskedChildContext=h),h}function Nn(n){return n=n.childContextTypes,n!=null}function Ra(){Gt(In),Gt(vn)}function _h(n,i,o){if(vn.current!==hr)throw Error(t(168));kt(vn,i),kt(In,o)}function gh(n,i,o){var c=n.stateNode;if(i=i.childContextTypes,typeof c.getChildContext!="function")return o;c=c.getChildContext();for(var h in c)if(!(h in i))throw Error(t(108,_e(n)||"Unknown",h));return ie({},o,c)}function Ca(n){return n=(n=n.stateNode)&&n.__reactInternalMemoizedMergedChildContext||hr,Or=vn.current,kt(vn,n),kt(In,In.current),!0}function vh(n,i,o){var c=n.stateNode;if(!c)throw Error(t(169));o?(n=gh(n,i,Or),c.__reactInternalMemoizedMergedChildContext=n,Gt(In),Gt(vn),kt(vn,n)):Gt(In),kt(In,o)}var ki=null,ba=!1,Nu=!1;function xh(n){ki===null?ki=[n]:ki.push(n)}function Gg(n){ba=!0,xh(n)}function pr(){if(!Nu&&ki!==null){Nu=!0;var n=0,i=_t;try{var o=ki;for(_t=1;n<o.length;n++){var c=o[n];do c=c(!0);while(c!==null)}ki=null,ba=!1}catch(h){throw ki!==null&&(ki=ki.slice(n+1)),ca(uo,pr),h}finally{_t=i,Nu=!1}}return null}var Ms=[],Es=0,Pa=null,La=0,jn=[],Kn=0,Br=null,zi=1,Hi="";function kr(n,i){Ms[Es++]=La,Ms[Es++]=Pa,Pa=n,La=i}function Sh(n,i,o){jn[Kn++]=zi,jn[Kn++]=Hi,jn[Kn++]=Br,Br=n;var c=zi;n=Hi;var h=32-Re(c)-1;c&=~(1<<h),o+=1;var x=32-Re(i)+h;if(30<x){var w=h-h%5;x=(c&(1<<w)-1).toString(32),c>>=w,h-=w,zi=1<<32-Re(i)+h|o<<h|c,Hi=x+n}else zi=1<<x|o<<h|c,Hi=n}function Uu(n){n.return!==null&&(kr(n,1),Sh(n,1,0))}function Fu(n){for(;n===Pa;)Pa=Ms[--Es],Ms[Es]=null,La=Ms[--Es],Ms[Es]=null;for(;n===Br;)Br=jn[--Kn],jn[Kn]=null,Hi=jn[--Kn],jn[Kn]=null,zi=jn[--Kn],jn[Kn]=null}var Vn=null,Gn=null,Xt=!1,li=null;function yh(n,i){var o=Jn(5,null,null,0);o.elementType="DELETED",o.stateNode=i,o.return=n,i=n.deletions,i===null?(n.deletions=[o],n.flags|=16):i.push(o)}function Mh(n,i){switch(n.tag){case 5:var o=n.type;return i=i.nodeType!==1||o.toLowerCase()!==i.nodeName.toLowerCase()?null:i,i!==null?(n.stateNode=i,Vn=n,Gn=fr(i.firstChild),!0):!1;case 6:return i=n.pendingProps===""||i.nodeType!==3?null:i,i!==null?(n.stateNode=i,Vn=n,Gn=null,!0):!1;case 13:return i=i.nodeType!==8?null:i,i!==null?(o=Br!==null?{id:zi,overflow:Hi}:null,n.memoizedState={dehydrated:i,treeContext:o,retryLane:1073741824},o=Jn(18,null,null,0),o.stateNode=i,o.return=n,n.child=o,Vn=n,Gn=null,!0):!1;default:return!1}}function Ou(n){return(n.mode&1)!==0&&(n.flags&128)===0}function Bu(n){if(Xt){var i=Gn;if(i){var o=i;if(!Mh(n,i)){if(Ou(n))throw Error(t(418));i=fr(o.nextSibling);var c=Vn;i&&Mh(n,i)?yh(c,o):(n.flags=n.flags&-4097|2,Xt=!1,Vn=n)}}else{if(Ou(n))throw Error(t(418));n.flags=n.flags&-4097|2,Xt=!1,Vn=n}}}function Eh(n){for(n=n.return;n!==null&&n.tag!==5&&n.tag!==3&&n.tag!==13;)n=n.return;Vn=n}function Da(n){if(n!==Vn)return!1;if(!Xt)return Eh(n),Xt=!0,!1;var i;if((i=n.tag!==3)&&!(i=n.tag!==5)&&(i=n.type,i=i!=="head"&&i!=="body"&&!bu(n.type,n.memoizedProps)),i&&(i=Gn)){if(Ou(n))throw Th(),Error(t(418));for(;i;)yh(n,i),i=fr(i.nextSibling)}if(Eh(n),n.tag===13){if(n=n.memoizedState,n=n!==null?n.dehydrated:null,!n)throw Error(t(317));e:{for(n=n.nextSibling,i=0;n;){if(n.nodeType===8){var o=n.data;if(o==="/$"){if(i===0){Gn=fr(n.nextSibling);break e}i--}else o!=="$"&&o!=="$!"&&o!=="$?"||i++}n=n.nextSibling}Gn=null}}else Gn=Vn?fr(n.stateNode.nextSibling):null;return!0}function Th(){for(var n=Gn;n;)n=fr(n.nextSibling)}function Ts(){Gn=Vn=null,Xt=!1}function ku(n){li===null?li=[n]:li.push(n)}var Wg=L.ReactCurrentBatchConfig;function Co(n,i,o){if(n=o.ref,n!==null&&typeof n!="function"&&typeof n!="object"){if(o._owner){if(o=o._owner,o){if(o.tag!==1)throw Error(t(309));var c=o.stateNode}if(!c)throw Error(t(147,n));var h=c,x=""+n;return i!==null&&i.ref!==null&&typeof i.ref=="function"&&i.ref._stringRef===x?i.ref:(i=function(w){var N=h.refs;w===null?delete N[x]:N[x]=w},i._stringRef=x,i)}if(typeof n!="string")throw Error(t(284));if(!o._owner)throw Error(t(290,n))}return n}function Ia(n,i){throw n=Object.prototype.toString.call(i),Error(t(31,n==="[object Object]"?"object with keys {"+Object.keys(i).join(", ")+"}":n))}function wh(n){var i=n._init;return i(n._payload)}function Ah(n){function i(Q,V){if(n){var ne=Q.deletions;ne===null?(Q.deletions=[V],Q.flags|=16):ne.push(V)}}function o(Q,V){if(!n)return null;for(;V!==null;)i(Q,V),V=V.sibling;return null}function c(Q,V){for(Q=new Map;V!==null;)V.key!==null?Q.set(V.key,V):Q.set(V.index,V),V=V.sibling;return Q}function h(Q,V){return Q=Mr(Q,V),Q.index=0,Q.sibling=null,Q}function x(Q,V,ne){return Q.index=ne,n?(ne=Q.alternate,ne!==null?(ne=ne.index,ne<V?(Q.flags|=2,V):ne):(Q.flags|=2,V)):(Q.flags|=1048576,V)}function w(Q){return n&&Q.alternate===null&&(Q.flags|=2),Q}function N(Q,V,ne,Ee){return V===null||V.tag!==6?(V=Pc(ne,Q.mode,Ee),V.return=Q,V):(V=h(V,ne),V.return=Q,V)}function B(Q,V,ne,Ee){var Ke=ne.type;return Ke===F?xe(Q,V,ne.props.children,Ee,ne.key):V!==null&&(V.elementType===Ke||typeof Ke=="object"&&Ke!==null&&Ke.$$typeof===Z&&wh(Ke)===V.type)?(Ee=h(V,ne.props),Ee.ref=Co(Q,V,ne),Ee.return=Q,Ee):(Ee=il(ne.type,ne.key,ne.props,null,Q.mode,Ee),Ee.ref=Co(Q,V,ne),Ee.return=Q,Ee)}function ue(Q,V,ne,Ee){return V===null||V.tag!==4||V.stateNode.containerInfo!==ne.containerInfo||V.stateNode.implementation!==ne.implementation?(V=Lc(ne,Q.mode,Ee),V.return=Q,V):(V=h(V,ne.children||[]),V.return=Q,V)}function xe(Q,V,ne,Ee,Ke){return V===null||V.tag!==7?(V=qr(ne,Q.mode,Ee,Ke),V.return=Q,V):(V=h(V,ne),V.return=Q,V)}function Se(Q,V,ne){if(typeof V=="string"&&V!==""||typeof V=="number")return V=Pc(""+V,Q.mode,ne),V.return=Q,V;if(typeof V=="object"&&V!==null){switch(V.$$typeof){case z:return ne=il(V.type,V.key,V.props,null,Q.mode,ne),ne.ref=Co(Q,null,V),ne.return=Q,ne;case D:return V=Lc(V,Q.mode,ne),V.return=Q,V;case Z:var Ee=V._init;return Se(Q,Ee(V._payload),ne)}if(H(V)||J(V))return V=qr(V,Q.mode,ne,null),V.return=Q,V;Ia(Q,V)}return null}function ve(Q,V,ne,Ee){var Ke=V!==null?V.key:null;if(typeof ne=="string"&&ne!==""||typeof ne=="number")return Ke!==null?null:N(Q,V,""+ne,Ee);if(typeof ne=="object"&&ne!==null){switch(ne.$$typeof){case z:return ne.key===Ke?B(Q,V,ne,Ee):null;case D:return ne.key===Ke?ue(Q,V,ne,Ee):null;case Z:return Ke=ne._init,ve(Q,V,Ke(ne._payload),Ee)}if(H(ne)||J(ne))return Ke!==null?null:xe(Q,V,ne,Ee,null);Ia(Q,ne)}return null}function ze(Q,V,ne,Ee,Ke){if(typeof Ee=="string"&&Ee!==""||typeof Ee=="number")return Q=Q.get(ne)||null,N(V,Q,""+Ee,Ke);if(typeof Ee=="object"&&Ee!==null){switch(Ee.$$typeof){case z:return Q=Q.get(Ee.key===null?ne:Ee.key)||null,B(V,Q,Ee,Ke);case D:return Q=Q.get(Ee.key===null?ne:Ee.key)||null,ue(V,Q,Ee,Ke);case Z:var Qe=Ee._init;return ze(Q,V,ne,Qe(Ee._payload),Ke)}if(H(Ee)||J(Ee))return Q=Q.get(ne)||null,xe(V,Q,Ee,Ke,null);Ia(V,Ee)}return null}function Ge(Q,V,ne,Ee){for(var Ke=null,Qe=null,Je=V,nt=V=0,fn=null;Je!==null&&nt<ne.length;nt++){Je.index>nt?(fn=Je,Je=null):fn=Je.sibling;var wt=ve(Q,Je,ne[nt],Ee);if(wt===null){Je===null&&(Je=fn);break}n&&Je&&wt.alternate===null&&i(Q,Je),V=x(wt,V,nt),Qe===null?Ke=wt:Qe.sibling=wt,Qe=wt,Je=fn}if(nt===ne.length)return o(Q,Je),Xt&&kr(Q,nt),Ke;if(Je===null){for(;nt<ne.length;nt++)Je=Se(Q,ne[nt],Ee),Je!==null&&(V=x(Je,V,nt),Qe===null?Ke=Je:Qe.sibling=Je,Qe=Je);return Xt&&kr(Q,nt),Ke}for(Je=c(Q,Je);nt<ne.length;nt++)fn=ze(Je,Q,nt,ne[nt],Ee),fn!==null&&(n&&fn.alternate!==null&&Je.delete(fn.key===null?nt:fn.key),V=x(fn,V,nt),Qe===null?Ke=fn:Qe.sibling=fn,Qe=fn);return n&&Je.forEach(function(Er){return i(Q,Er)}),Xt&&kr(Q,nt),Ke}function Ye(Q,V,ne,Ee){var Ke=J(ne);if(typeof Ke!="function")throw Error(t(150));if(ne=Ke.call(ne),ne==null)throw Error(t(151));for(var Qe=Ke=null,Je=V,nt=V=0,fn=null,wt=ne.next();Je!==null&&!wt.done;nt++,wt=ne.next()){Je.index>nt?(fn=Je,Je=null):fn=Je.sibling;var Er=ve(Q,Je,wt.value,Ee);if(Er===null){Je===null&&(Je=fn);break}n&&Je&&Er.alternate===null&&i(Q,Je),V=x(Er,V,nt),Qe===null?Ke=Er:Qe.sibling=Er,Qe=Er,Je=fn}if(wt.done)return o(Q,Je),Xt&&kr(Q,nt),Ke;if(Je===null){for(;!wt.done;nt++,wt=ne.next())wt=Se(Q,wt.value,Ee),wt!==null&&(V=x(wt,V,nt),Qe===null?Ke=wt:Qe.sibling=wt,Qe=wt);return Xt&&kr(Q,nt),Ke}for(Je=c(Q,Je);!wt.done;nt++,wt=ne.next())wt=ze(Je,Q,nt,wt.value,Ee),wt!==null&&(n&&wt.alternate!==null&&Je.delete(wt.key===null?nt:wt.key),V=x(wt,V,nt),Qe===null?Ke=wt:Qe.sibling=wt,Qe=wt);return n&&Je.forEach(function(E0){return i(Q,E0)}),Xt&&kr(Q,nt),Ke}function Qt(Q,V,ne,Ee){if(typeof ne=="object"&&ne!==null&&ne.type===F&&ne.key===null&&(ne=ne.props.children),typeof ne=="object"&&ne!==null){switch(ne.$$typeof){case z:e:{for(var Ke=ne.key,Qe=V;Qe!==null;){if(Qe.key===Ke){if(Ke=ne.type,Ke===F){if(Qe.tag===7){o(Q,Qe.sibling),V=h(Qe,ne.props.children),V.return=Q,Q=V;break e}}else if(Qe.elementType===Ke||typeof Ke=="object"&&Ke!==null&&Ke.$$typeof===Z&&wh(Ke)===Qe.type){o(Q,Qe.sibling),V=h(Qe,ne.props),V.ref=Co(Q,Qe,ne),V.return=Q,Q=V;break e}o(Q,Qe);break}else i(Q,Qe);Qe=Qe.sibling}ne.type===F?(V=qr(ne.props.children,Q.mode,Ee,ne.key),V.return=Q,Q=V):(Ee=il(ne.type,ne.key,ne.props,null,Q.mode,Ee),Ee.ref=Co(Q,V,ne),Ee.return=Q,Q=Ee)}return w(Q);case D:e:{for(Qe=ne.key;V!==null;){if(V.key===Qe)if(V.tag===4&&V.stateNode.containerInfo===ne.containerInfo&&V.stateNode.implementation===ne.implementation){o(Q,V.sibling),V=h(V,ne.children||[]),V.return=Q,Q=V;break e}else{o(Q,V);break}else i(Q,V);V=V.sibling}V=Lc(ne,Q.mode,Ee),V.return=Q,Q=V}return w(Q);case Z:return Qe=ne._init,Qt(Q,V,Qe(ne._payload),Ee)}if(H(ne))return Ge(Q,V,ne,Ee);if(J(ne))return Ye(Q,V,ne,Ee);Ia(Q,ne)}return typeof ne=="string"&&ne!==""||typeof ne=="number"?(ne=""+ne,V!==null&&V.tag===6?(o(Q,V.sibling),V=h(V,ne),V.return=Q,Q=V):(o(Q,V),V=Pc(ne,Q.mode,Ee),V.return=Q,Q=V),w(Q)):o(Q,V)}return Qt}var ws=Ah(!0),Rh=Ah(!1),Na=dr(null),Ua=null,As=null,zu=null;function Hu(){zu=As=Ua=null}function Vu(n){var i=Na.current;Gt(Na),n._currentValue=i}function Gu(n,i,o){for(;n!==null;){var c=n.alternate;if((n.childLanes&i)!==i?(n.childLanes|=i,c!==null&&(c.childLanes|=i)):c!==null&&(c.childLanes&i)!==i&&(c.childLanes|=i),n===o)break;n=n.return}}function Rs(n,i){Ua=n,zu=As=null,n=n.dependencies,n!==null&&n.firstContext!==null&&((n.lanes&i)!==0&&(Un=!0),n.firstContext=null)}function $n(n){var i=n._currentValue;if(zu!==n)if(n={context:n,memoizedValue:i,next:null},As===null){if(Ua===null)throw Error(t(308));As=n,Ua.dependencies={lanes:0,firstContext:n}}else As=As.next=n;return i}var zr=null;function Wu(n){zr===null?zr=[n]:zr.push(n)}function Ch(n,i,o,c){var h=i.interleaved;return h===null?(o.next=o,Wu(i)):(o.next=h.next,h.next=o),i.interleaved=o,Vi(n,c)}function Vi(n,i){n.lanes|=i;var o=n.alternate;for(o!==null&&(o.lanes|=i),o=n,n=n.return;n!==null;)n.childLanes|=i,o=n.alternate,o!==null&&(o.childLanes|=i),o=n,n=n.return;return o.tag===3?o.stateNode:null}var mr=!1;function Xu(n){n.updateQueue={baseState:n.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function bh(n,i){n=n.updateQueue,i.updateQueue===n&&(i.updateQueue={baseState:n.baseState,firstBaseUpdate:n.firstBaseUpdate,lastBaseUpdate:n.lastBaseUpdate,shared:n.shared,effects:n.effects})}function Gi(n,i){return{eventTime:n,lane:i,tag:0,payload:null,callback:null,next:null}}function _r(n,i,o){var c=n.updateQueue;if(c===null)return null;if(c=c.shared,(yt&2)!==0){var h=c.pending;return h===null?i.next=i:(i.next=h.next,h.next=i),c.pending=i,Vi(n,o)}return h=c.interleaved,h===null?(i.next=i,Wu(c)):(i.next=h.next,h.next=i),c.interleaved=i,Vi(n,o)}function Fa(n,i,o){if(i=i.updateQueue,i!==null&&(i=i.shared,(o&4194240)!==0)){var c=i.lanes;c&=n.pendingLanes,o|=c,i.lanes=o,Dn(n,o)}}function Ph(n,i){var o=n.updateQueue,c=n.alternate;if(c!==null&&(c=c.updateQueue,o===c)){var h=null,x=null;if(o=o.firstBaseUpdate,o!==null){do{var w={eventTime:o.eventTime,lane:o.lane,tag:o.tag,payload:o.payload,callback:o.callback,next:null};x===null?h=x=w:x=x.next=w,o=o.next}while(o!==null);x===null?h=x=i:x=x.next=i}else h=x=i;o={baseState:c.baseState,firstBaseUpdate:h,lastBaseUpdate:x,shared:c.shared,effects:c.effects},n.updateQueue=o;return}n=o.lastBaseUpdate,n===null?o.firstBaseUpdate=i:n.next=i,o.lastBaseUpdate=i}function Oa(n,i,o,c){var h=n.updateQueue;mr=!1;var x=h.firstBaseUpdate,w=h.lastBaseUpdate,N=h.shared.pending;if(N!==null){h.shared.pending=null;var B=N,ue=B.next;B.next=null,w===null?x=ue:w.next=ue,w=B;var xe=n.alternate;xe!==null&&(xe=xe.updateQueue,N=xe.lastBaseUpdate,N!==w&&(N===null?xe.firstBaseUpdate=ue:N.next=ue,xe.lastBaseUpdate=B))}if(x!==null){var Se=h.baseState;w=0,xe=ue=B=null,N=x;do{var ve=N.lane,ze=N.eventTime;if((c&ve)===ve){xe!==null&&(xe=xe.next={eventTime:ze,lane:0,tag:N.tag,payload:N.payload,callback:N.callback,next:null});e:{var Ge=n,Ye=N;switch(ve=i,ze=o,Ye.tag){case 1:if(Ge=Ye.payload,typeof Ge=="function"){Se=Ge.call(ze,Se,ve);break e}Se=Ge;break e;case 3:Ge.flags=Ge.flags&-65537|128;case 0:if(Ge=Ye.payload,ve=typeof Ge=="function"?Ge.call(ze,Se,ve):Ge,ve==null)break e;Se=ie({},Se,ve);break e;case 2:mr=!0}}N.callback!==null&&N.lane!==0&&(n.flags|=64,ve=h.effects,ve===null?h.effects=[N]:ve.push(N))}else ze={eventTime:ze,lane:ve,tag:N.tag,payload:N.payload,callback:N.callback,next:null},xe===null?(ue=xe=ze,B=Se):xe=xe.next=ze,w|=ve;if(N=N.next,N===null){if(N=h.shared.pending,N===null)break;ve=N,N=ve.next,ve.next=null,h.lastBaseUpdate=ve,h.shared.pending=null}}while(!0);if(xe===null&&(B=Se),h.baseState=B,h.firstBaseUpdate=ue,h.lastBaseUpdate=xe,i=h.shared.interleaved,i!==null){h=i;do w|=h.lane,h=h.next;while(h!==i)}else x===null&&(h.shared.lanes=0);Gr|=w,n.lanes=w,n.memoizedState=Se}}function Lh(n,i,o){if(n=i.effects,i.effects=null,n!==null)for(i=0;i<n.length;i++){var c=n[i],h=c.callback;if(h!==null){if(c.callback=null,c=o,typeof h!="function")throw Error(t(191,h));h.call(c)}}}var bo={},Ei=dr(bo),Po=dr(bo),Lo=dr(bo);function Hr(n){if(n===bo)throw Error(t(174));return n}function Yu(n,i){switch(kt(Lo,i),kt(Po,n),kt(Ei,bo),n=i.nodeType,n){case 9:case 11:i=(i=i.documentElement)?i.namespaceURI:T(null,"");break;default:n=n===8?i.parentNode:i,i=n.namespaceURI||null,n=n.tagName,i=T(i,n)}Gt(Ei),kt(Ei,i)}function Cs(){Gt(Ei),Gt(Po),Gt(Lo)}function Dh(n){Hr(Lo.current);var i=Hr(Ei.current),o=T(i,n.type);i!==o&&(kt(Po,n),kt(Ei,o))}function qu(n){Po.current===n&&(Gt(Ei),Gt(Po))}var jt=dr(0);function Ba(n){for(var i=n;i!==null;){if(i.tag===13){var o=i.memoizedState;if(o!==null&&(o=o.dehydrated,o===null||o.data==="$?"||o.data==="$!"))return i}else if(i.tag===19&&i.memoizedProps.revealOrder!==void 0){if((i.flags&128)!==0)return i}else if(i.child!==null){i.child.return=i,i=i.child;continue}if(i===n)break;for(;i.sibling===null;){if(i.return===null||i.return===n)return null;i=i.return}i.sibling.return=i.return,i=i.sibling}return null}var ju=[];function Ku(){for(var n=0;n<ju.length;n++)ju[n]._workInProgressVersionPrimary=null;ju.length=0}var ka=L.ReactCurrentDispatcher,$u=L.ReactCurrentBatchConfig,Vr=0,Kt=null,sn=null,un=null,za=!1,Do=!1,Io=0,Xg=0;function xn(){throw Error(t(321))}function Zu(n,i){if(i===null)return!1;for(var o=0;o<i.length&&o<n.length;o++)if(!ai(n[o],i[o]))return!1;return!0}function Qu(n,i,o,c,h,x){if(Vr=x,Kt=i,i.memoizedState=null,i.updateQueue=null,i.lanes=0,ka.current=n===null||n.memoizedState===null?Kg:$g,n=o(c,h),Do){x=0;do{if(Do=!1,Io=0,25<=x)throw Error(t(301));x+=1,un=sn=null,i.updateQueue=null,ka.current=Zg,n=o(c,h)}while(Do)}if(ka.current=Ga,i=sn!==null&&sn.next!==null,Vr=0,un=sn=Kt=null,za=!1,i)throw Error(t(300));return n}function Ju(){var n=Io!==0;return Io=0,n}function Ti(){var n={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return un===null?Kt.memoizedState=un=n:un=un.next=n,un}function Zn(){if(sn===null){var n=Kt.alternate;n=n!==null?n.memoizedState:null}else n=sn.next;var i=un===null?Kt.memoizedState:un.next;if(i!==null)un=i,sn=n;else{if(n===null)throw Error(t(310));sn=n,n={memoizedState:sn.memoizedState,baseState:sn.baseState,baseQueue:sn.baseQueue,queue:sn.queue,next:null},un===null?Kt.memoizedState=un=n:un=un.next=n}return un}function No(n,i){return typeof i=="function"?i(n):i}function ec(n){var i=Zn(),o=i.queue;if(o===null)throw Error(t(311));o.lastRenderedReducer=n;var c=sn,h=c.baseQueue,x=o.pending;if(x!==null){if(h!==null){var w=h.next;h.next=x.next,x.next=w}c.baseQueue=h=x,o.pending=null}if(h!==null){x=h.next,c=c.baseState;var N=w=null,B=null,ue=x;do{var xe=ue.lane;if((Vr&xe)===xe)B!==null&&(B=B.next={lane:0,action:ue.action,hasEagerState:ue.hasEagerState,eagerState:ue.eagerState,next:null}),c=ue.hasEagerState?ue.eagerState:n(c,ue.action);else{var Se={lane:xe,action:ue.action,hasEagerState:ue.hasEagerState,eagerState:ue.eagerState,next:null};B===null?(N=B=Se,w=c):B=B.next=Se,Kt.lanes|=xe,Gr|=xe}ue=ue.next}while(ue!==null&&ue!==x);B===null?w=c:B.next=N,ai(c,i.memoizedState)||(Un=!0),i.memoizedState=c,i.baseState=w,i.baseQueue=B,o.lastRenderedState=c}if(n=o.interleaved,n!==null){h=n;do x=h.lane,Kt.lanes|=x,Gr|=x,h=h.next;while(h!==n)}else h===null&&(o.lanes=0);return[i.memoizedState,o.dispatch]}function tc(n){var i=Zn(),o=i.queue;if(o===null)throw Error(t(311));o.lastRenderedReducer=n;var c=o.dispatch,h=o.pending,x=i.memoizedState;if(h!==null){o.pending=null;var w=h=h.next;do x=n(x,w.action),w=w.next;while(w!==h);ai(x,i.memoizedState)||(Un=!0),i.memoizedState=x,i.baseQueue===null&&(i.baseState=x),o.lastRenderedState=x}return[x,c]}function Ih(){}function Nh(n,i){var o=Kt,c=Zn(),h=i(),x=!ai(c.memoizedState,h);if(x&&(c.memoizedState=h,Un=!0),c=c.queue,nc(Oh.bind(null,o,c,n),[n]),c.getSnapshot!==i||x||un!==null&&un.memoizedState.tag&1){if(o.flags|=2048,Uo(9,Fh.bind(null,o,c,h,i),void 0,null),cn===null)throw Error(t(349));(Vr&30)!==0||Uh(o,i,h)}return h}function Uh(n,i,o){n.flags|=16384,n={getSnapshot:i,value:o},i=Kt.updateQueue,i===null?(i={lastEffect:null,stores:null},Kt.updateQueue=i,i.stores=[n]):(o=i.stores,o===null?i.stores=[n]:o.push(n))}function Fh(n,i,o,c){i.value=o,i.getSnapshot=c,Bh(i)&&kh(n)}function Oh(n,i,o){return o(function(){Bh(i)&&kh(n)})}function Bh(n){var i=n.getSnapshot;n=n.value;try{var o=i();return!ai(n,o)}catch{return!0}}function kh(n){var i=Vi(n,1);i!==null&&di(i,n,1,-1)}function zh(n){var i=Ti();return typeof n=="function"&&(n=n()),i.memoizedState=i.baseState=n,n={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:No,lastRenderedState:n},i.queue=n,n=n.dispatch=jg.bind(null,Kt,n),[i.memoizedState,n]}function Uo(n,i,o,c){return n={tag:n,create:i,destroy:o,deps:c,next:null},i=Kt.updateQueue,i===null?(i={lastEffect:null,stores:null},Kt.updateQueue=i,i.lastEffect=n.next=n):(o=i.lastEffect,o===null?i.lastEffect=n.next=n:(c=o.next,o.next=n,n.next=c,i.lastEffect=n)),n}function Hh(){return Zn().memoizedState}function Ha(n,i,o,c){var h=Ti();Kt.flags|=n,h.memoizedState=Uo(1|i,o,void 0,c===void 0?null:c)}function Va(n,i,o,c){var h=Zn();c=c===void 0?null:c;var x=void 0;if(sn!==null){var w=sn.memoizedState;if(x=w.destroy,c!==null&&Zu(c,w.deps)){h.memoizedState=Uo(i,o,x,c);return}}Kt.flags|=n,h.memoizedState=Uo(1|i,o,x,c)}function Vh(n,i){return Ha(8390656,8,n,i)}function nc(n,i){return Va(2048,8,n,i)}function Gh(n,i){return Va(4,2,n,i)}function Wh(n,i){return Va(4,4,n,i)}function Xh(n,i){if(typeof i=="function")return n=n(),i(n),function(){i(null)};if(i!=null)return n=n(),i.current=n,function(){i.current=null}}function Yh(n,i,o){return o=o!=null?o.concat([n]):null,Va(4,4,Xh.bind(null,i,n),o)}function ic(){}function qh(n,i){var o=Zn();i=i===void 0?null:i;var c=o.memoizedState;return c!==null&&i!==null&&Zu(i,c[1])?c[0]:(o.memoizedState=[n,i],n)}function jh(n,i){var o=Zn();i=i===void 0?null:i;var c=o.memoizedState;return c!==null&&i!==null&&Zu(i,c[1])?c[0]:(n=n(),o.memoizedState=[n,i],n)}function Kh(n,i,o){return(Vr&21)===0?(n.baseState&&(n.baseState=!1,Un=!0),n.memoizedState=o):(ai(o,i)||(o=ke(),Kt.lanes|=o,Gr|=o,n.baseState=!0),i)}function Yg(n,i){var o=_t;_t=o!==0&&4>o?o:4,n(!0);var c=$u.transition;$u.transition={};try{n(!1),i()}finally{_t=o,$u.transition=c}}function $h(){return Zn().memoizedState}function qg(n,i,o){var c=Sr(n);if(o={lane:c,action:o,hasEagerState:!1,eagerState:null,next:null},Zh(n))Qh(i,o);else if(o=Ch(n,i,o,c),o!==null){var h=Rn();di(o,n,c,h),Jh(o,i,c)}}function jg(n,i,o){var c=Sr(n),h={lane:c,action:o,hasEagerState:!1,eagerState:null,next:null};if(Zh(n))Qh(i,h);else{var x=n.alternate;if(n.lanes===0&&(x===null||x.lanes===0)&&(x=i.lastRenderedReducer,x!==null))try{var w=i.lastRenderedState,N=x(w,o);if(h.hasEagerState=!0,h.eagerState=N,ai(N,w)){var B=i.interleaved;B===null?(h.next=h,Wu(i)):(h.next=B.next,B.next=h),i.interleaved=h;return}}catch{}finally{}o=Ch(n,i,h,c),o!==null&&(h=Rn(),di(o,n,c,h),Jh(o,i,c))}}function Zh(n){var i=n.alternate;return n===Kt||i!==null&&i===Kt}function Qh(n,i){Do=za=!0;var o=n.pending;o===null?i.next=i:(i.next=o.next,o.next=i),n.pending=i}function Jh(n,i,o){if((o&4194240)!==0){var c=i.lanes;c&=n.pendingLanes,o|=c,i.lanes=o,Dn(n,o)}}var Ga={readContext:$n,useCallback:xn,useContext:xn,useEffect:xn,useImperativeHandle:xn,useInsertionEffect:xn,useLayoutEffect:xn,useMemo:xn,useReducer:xn,useRef:xn,useState:xn,useDebugValue:xn,useDeferredValue:xn,useTransition:xn,useMutableSource:xn,useSyncExternalStore:xn,useId:xn,unstable_isNewReconciler:!1},Kg={readContext:$n,useCallback:function(n,i){return Ti().memoizedState=[n,i===void 0?null:i],n},useContext:$n,useEffect:Vh,useImperativeHandle:function(n,i,o){return o=o!=null?o.concat([n]):null,Ha(4194308,4,Xh.bind(null,i,n),o)},useLayoutEffect:function(n,i){return Ha(4194308,4,n,i)},useInsertionEffect:function(n,i){return Ha(4,2,n,i)},useMemo:function(n,i){var o=Ti();return i=i===void 0?null:i,n=n(),o.memoizedState=[n,i],n},useReducer:function(n,i,o){var c=Ti();return i=o!==void 0?o(i):i,c.memoizedState=c.baseState=i,n={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:n,lastRenderedState:i},c.queue=n,n=n.dispatch=qg.bind(null,Kt,n),[c.memoizedState,n]},useRef:function(n){var i=Ti();return n={current:n},i.memoizedState=n},useState:zh,useDebugValue:ic,useDeferredValue:function(n){return Ti().memoizedState=n},useTransition:function(){var n=zh(!1),i=n[0];return n=Yg.bind(null,n[1]),Ti().memoizedState=n,[i,n]},useMutableSource:function(){},useSyncExternalStore:function(n,i,o){var c=Kt,h=Ti();if(Xt){if(o===void 0)throw Error(t(407));o=o()}else{if(o=i(),cn===null)throw Error(t(349));(Vr&30)!==0||Uh(c,i,o)}h.memoizedState=o;var x={value:o,getSnapshot:i};return h.queue=x,Vh(Oh.bind(null,c,x,n),[n]),c.flags|=2048,Uo(9,Fh.bind(null,c,x,o,i),void 0,null),o},useId:function(){var n=Ti(),i=cn.identifierPrefix;if(Xt){var o=Hi,c=zi;o=(c&~(1<<32-Re(c)-1)).toString(32)+o,i=":"+i+"R"+o,o=Io++,0<o&&(i+="H"+o.toString(32)),i+=":"}else o=Xg++,i=":"+i+"r"+o.toString(32)+":";return n.memoizedState=i},unstable_isNewReconciler:!1},$g={readContext:$n,useCallback:qh,useContext:$n,useEffect:nc,useImperativeHandle:Yh,useInsertionEffect:Gh,useLayoutEffect:Wh,useMemo:jh,useReducer:ec,useRef:Hh,useState:function(){return ec(No)},useDebugValue:ic,useDeferredValue:function(n){var i=Zn();return Kh(i,sn.memoizedState,n)},useTransition:function(){var n=ec(No)[0],i=Zn().memoizedState;return[n,i]},useMutableSource:Ih,useSyncExternalStore:Nh,useId:$h,unstable_isNewReconciler:!1},Zg={readContext:$n,useCallback:qh,useContext:$n,useEffect:nc,useImperativeHandle:Yh,useInsertionEffect:Gh,useLayoutEffect:Wh,useMemo:jh,useReducer:tc,useRef:Hh,useState:function(){return tc(No)},useDebugValue:ic,useDeferredValue:function(n){var i=Zn();return sn===null?i.memoizedState=n:Kh(i,sn.memoizedState,n)},useTransition:function(){var n=tc(No)[0],i=Zn().memoizedState;return[n,i]},useMutableSource:Ih,useSyncExternalStore:Nh,useId:$h,unstable_isNewReconciler:!1};function ui(n,i){if(n&&n.defaultProps){i=ie({},i),n=n.defaultProps;for(var o in n)i[o]===void 0&&(i[o]=n[o]);return i}return i}function rc(n,i,o,c){i=n.memoizedState,o=o(c,i),o=o==null?i:ie({},i,o),n.memoizedState=o,n.lanes===0&&(n.updateQueue.baseState=o)}var Wa={isMounted:function(n){return(n=n._reactInternals)?Si(n)===n:!1},enqueueSetState:function(n,i,o){n=n._reactInternals;var c=Rn(),h=Sr(n),x=Gi(c,h);x.payload=i,o!=null&&(x.callback=o),i=_r(n,x,h),i!==null&&(di(i,n,h,c),Fa(i,n,h))},enqueueReplaceState:function(n,i,o){n=n._reactInternals;var c=Rn(),h=Sr(n),x=Gi(c,h);x.tag=1,x.payload=i,o!=null&&(x.callback=o),i=_r(n,x,h),i!==null&&(di(i,n,h,c),Fa(i,n,h))},enqueueForceUpdate:function(n,i){n=n._reactInternals;var o=Rn(),c=Sr(n),h=Gi(o,c);h.tag=2,i!=null&&(h.callback=i),i=_r(n,h,c),i!==null&&(di(i,n,c,o),Fa(i,n,c))}};function ep(n,i,o,c,h,x,w){return n=n.stateNode,typeof n.shouldComponentUpdate=="function"?n.shouldComponentUpdate(c,x,w):i.prototype&&i.prototype.isPureReactComponent?!yo(o,c)||!yo(h,x):!0}function tp(n,i,o){var c=!1,h=hr,x=i.contextType;return typeof x=="object"&&x!==null?x=$n(x):(h=Nn(i)?Or:vn.current,c=i.contextTypes,x=(c=c!=null)?ys(n,h):hr),i=new i(o,x),n.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,i.updater=Wa,n.stateNode=i,i._reactInternals=n,c&&(n=n.stateNode,n.__reactInternalMemoizedUnmaskedChildContext=h,n.__reactInternalMemoizedMaskedChildContext=x),i}function np(n,i,o,c){n=i.state,typeof i.componentWillReceiveProps=="function"&&i.componentWillReceiveProps(o,c),typeof i.UNSAFE_componentWillReceiveProps=="function"&&i.UNSAFE_componentWillReceiveProps(o,c),i.state!==n&&Wa.enqueueReplaceState(i,i.state,null)}function sc(n,i,o,c){var h=n.stateNode;h.props=o,h.state=n.memoizedState,h.refs={},Xu(n);var x=i.contextType;typeof x=="object"&&x!==null?h.context=$n(x):(x=Nn(i)?Or:vn.current,h.context=ys(n,x)),h.state=n.memoizedState,x=i.getDerivedStateFromProps,typeof x=="function"&&(rc(n,i,x,o),h.state=n.memoizedState),typeof i.getDerivedStateFromProps=="function"||typeof h.getSnapshotBeforeUpdate=="function"||typeof h.UNSAFE_componentWillMount!="function"&&typeof h.componentWillMount!="function"||(i=h.state,typeof h.componentWillMount=="function"&&h.componentWillMount(),typeof h.UNSAFE_componentWillMount=="function"&&h.UNSAFE_componentWillMount(),i!==h.state&&Wa.enqueueReplaceState(h,h.state,null),Oa(n,o,h,c),h.state=n.memoizedState),typeof h.componentDidMount=="function"&&(n.flags|=4194308)}function bs(n,i){try{var o="",c=i;do o+=we(c),c=c.return;while(c);var h=o}catch(x){h=`
Error generating stack: `+x.message+`
`+x.stack}return{value:n,source:i,stack:h,digest:null}}function oc(n,i,o){return{value:n,source:null,stack:o??null,digest:i??null}}function ac(n,i){try{console.error(i.value)}catch(o){setTimeout(function(){throw o})}}var Qg=typeof WeakMap=="function"?WeakMap:Map;function ip(n,i,o){o=Gi(-1,o),o.tag=3,o.payload={element:null};var c=i.value;return o.callback=function(){Za||(Za=!0,Mc=c),ac(n,i)},o}function rp(n,i,o){o=Gi(-1,o),o.tag=3;var c=n.type.getDerivedStateFromError;if(typeof c=="function"){var h=i.value;o.payload=function(){return c(h)},o.callback=function(){ac(n,i)}}var x=n.stateNode;return x!==null&&typeof x.componentDidCatch=="function"&&(o.callback=function(){ac(n,i),typeof c!="function"&&(vr===null?vr=new Set([this]):vr.add(this));var w=i.stack;this.componentDidCatch(i.value,{componentStack:w!==null?w:""})}),o}function sp(n,i,o){var c=n.pingCache;if(c===null){c=n.pingCache=new Qg;var h=new Set;c.set(i,h)}else h=c.get(i),h===void 0&&(h=new Set,c.set(i,h));h.has(o)||(h.add(o),n=d0.bind(null,n,i,o),i.then(n,n))}function op(n){do{var i;if((i=n.tag===13)&&(i=n.memoizedState,i=i!==null?i.dehydrated!==null:!0),i)return n;n=n.return}while(n!==null);return null}function ap(n,i,o,c,h){return(n.mode&1)===0?(n===i?n.flags|=65536:(n.flags|=128,o.flags|=131072,o.flags&=-52805,o.tag===1&&(o.alternate===null?o.tag=17:(i=Gi(-1,1),i.tag=2,_r(o,i,1))),o.lanes|=1),n):(n.flags|=65536,n.lanes=h,n)}var Jg=L.ReactCurrentOwner,Un=!1;function An(n,i,o,c){i.child=n===null?Rh(i,null,o,c):ws(i,n.child,o,c)}function lp(n,i,o,c,h){o=o.render;var x=i.ref;return Rs(i,h),c=Qu(n,i,o,c,x,h),o=Ju(),n!==null&&!Un?(i.updateQueue=n.updateQueue,i.flags&=-2053,n.lanes&=~h,Wi(n,i,h)):(Xt&&o&&Uu(i),i.flags|=1,An(n,i,c,h),i.child)}function up(n,i,o,c,h){if(n===null){var x=o.type;return typeof x=="function"&&!bc(x)&&x.defaultProps===void 0&&o.compare===null&&o.defaultProps===void 0?(i.tag=15,i.type=x,cp(n,i,x,c,h)):(n=il(o.type,null,c,i,i.mode,h),n.ref=i.ref,n.return=i,i.child=n)}if(x=n.child,(n.lanes&h)===0){var w=x.memoizedProps;if(o=o.compare,o=o!==null?o:yo,o(w,c)&&n.ref===i.ref)return Wi(n,i,h)}return i.flags|=1,n=Mr(x,c),n.ref=i.ref,n.return=i,i.child=n}function cp(n,i,o,c,h){if(n!==null){var x=n.memoizedProps;if(yo(x,c)&&n.ref===i.ref)if(Un=!1,i.pendingProps=c=x,(n.lanes&h)!==0)(n.flags&131072)!==0&&(Un=!0);else return i.lanes=n.lanes,Wi(n,i,h)}return lc(n,i,o,c,h)}function fp(n,i,o){var c=i.pendingProps,h=c.children,x=n!==null?n.memoizedState:null;if(c.mode==="hidden")if((i.mode&1)===0)i.memoizedState={baseLanes:0,cachePool:null,transitions:null},kt(Ls,Wn),Wn|=o;else{if((o&1073741824)===0)return n=x!==null?x.baseLanes|o:o,i.lanes=i.childLanes=1073741824,i.memoizedState={baseLanes:n,cachePool:null,transitions:null},i.updateQueue=null,kt(Ls,Wn),Wn|=n,null;i.memoizedState={baseLanes:0,cachePool:null,transitions:null},c=x!==null?x.baseLanes:o,kt(Ls,Wn),Wn|=c}else x!==null?(c=x.baseLanes|o,i.memoizedState=null):c=o,kt(Ls,Wn),Wn|=c;return An(n,i,h,o),i.child}function dp(n,i){var o=i.ref;(n===null&&o!==null||n!==null&&n.ref!==o)&&(i.flags|=512,i.flags|=2097152)}function lc(n,i,o,c,h){var x=Nn(o)?Or:vn.current;return x=ys(i,x),Rs(i,h),o=Qu(n,i,o,c,x,h),c=Ju(),n!==null&&!Un?(i.updateQueue=n.updateQueue,i.flags&=-2053,n.lanes&=~h,Wi(n,i,h)):(Xt&&c&&Uu(i),i.flags|=1,An(n,i,o,h),i.child)}function hp(n,i,o,c,h){if(Nn(o)){var x=!0;Ca(i)}else x=!1;if(Rs(i,h),i.stateNode===null)Ya(n,i),tp(i,o,c),sc(i,o,c,h),c=!0;else if(n===null){var w=i.stateNode,N=i.memoizedProps;w.props=N;var B=w.context,ue=o.contextType;typeof ue=="object"&&ue!==null?ue=$n(ue):(ue=Nn(o)?Or:vn.current,ue=ys(i,ue));var xe=o.getDerivedStateFromProps,Se=typeof xe=="function"||typeof w.getSnapshotBeforeUpdate=="function";Se||typeof w.UNSAFE_componentWillReceiveProps!="function"&&typeof w.componentWillReceiveProps!="function"||(N!==c||B!==ue)&&np(i,w,c,ue),mr=!1;var ve=i.memoizedState;w.state=ve,Oa(i,c,w,h),B=i.memoizedState,N!==c||ve!==B||In.current||mr?(typeof xe=="function"&&(rc(i,o,xe,c),B=i.memoizedState),(N=mr||ep(i,o,N,c,ve,B,ue))?(Se||typeof w.UNSAFE_componentWillMount!="function"&&typeof w.componentWillMount!="function"||(typeof w.componentWillMount=="function"&&w.componentWillMount(),typeof w.UNSAFE_componentWillMount=="function"&&w.UNSAFE_componentWillMount()),typeof w.componentDidMount=="function"&&(i.flags|=4194308)):(typeof w.componentDidMount=="function"&&(i.flags|=4194308),i.memoizedProps=c,i.memoizedState=B),w.props=c,w.state=B,w.context=ue,c=N):(typeof w.componentDidMount=="function"&&(i.flags|=4194308),c=!1)}else{w=i.stateNode,bh(n,i),N=i.memoizedProps,ue=i.type===i.elementType?N:ui(i.type,N),w.props=ue,Se=i.pendingProps,ve=w.context,B=o.contextType,typeof B=="object"&&B!==null?B=$n(B):(B=Nn(o)?Or:vn.current,B=ys(i,B));var ze=o.getDerivedStateFromProps;(xe=typeof ze=="function"||typeof w.getSnapshotBeforeUpdate=="function")||typeof w.UNSAFE_componentWillReceiveProps!="function"&&typeof w.componentWillReceiveProps!="function"||(N!==Se||ve!==B)&&np(i,w,c,B),mr=!1,ve=i.memoizedState,w.state=ve,Oa(i,c,w,h);var Ge=i.memoizedState;N!==Se||ve!==Ge||In.current||mr?(typeof ze=="function"&&(rc(i,o,ze,c),Ge=i.memoizedState),(ue=mr||ep(i,o,ue,c,ve,Ge,B)||!1)?(xe||typeof w.UNSAFE_componentWillUpdate!="function"&&typeof w.componentWillUpdate!="function"||(typeof w.componentWillUpdate=="function"&&w.componentWillUpdate(c,Ge,B),typeof w.UNSAFE_componentWillUpdate=="function"&&w.UNSAFE_componentWillUpdate(c,Ge,B)),typeof w.componentDidUpdate=="function"&&(i.flags|=4),typeof w.getSnapshotBeforeUpdate=="function"&&(i.flags|=1024)):(typeof w.componentDidUpdate!="function"||N===n.memoizedProps&&ve===n.memoizedState||(i.flags|=4),typeof w.getSnapshotBeforeUpdate!="function"||N===n.memoizedProps&&ve===n.memoizedState||(i.flags|=1024),i.memoizedProps=c,i.memoizedState=Ge),w.props=c,w.state=Ge,w.context=B,c=ue):(typeof w.componentDidUpdate!="function"||N===n.memoizedProps&&ve===n.memoizedState||(i.flags|=4),typeof w.getSnapshotBeforeUpdate!="function"||N===n.memoizedProps&&ve===n.memoizedState||(i.flags|=1024),c=!1)}return uc(n,i,o,c,x,h)}function uc(n,i,o,c,h,x){dp(n,i);var w=(i.flags&128)!==0;if(!c&&!w)return h&&vh(i,o,!1),Wi(n,i,x);c=i.stateNode,Jg.current=i;var N=w&&typeof o.getDerivedStateFromError!="function"?null:c.render();return i.flags|=1,n!==null&&w?(i.child=ws(i,n.child,null,x),i.child=ws(i,null,N,x)):An(n,i,N,x),i.memoizedState=c.state,h&&vh(i,o,!0),i.child}function pp(n){var i=n.stateNode;i.pendingContext?_h(n,i.pendingContext,i.pendingContext!==i.context):i.context&&_h(n,i.context,!1),Yu(n,i.containerInfo)}function mp(n,i,o,c,h){return Ts(),ku(h),i.flags|=256,An(n,i,o,c),i.child}var cc={dehydrated:null,treeContext:null,retryLane:0};function fc(n){return{baseLanes:n,cachePool:null,transitions:null}}function _p(n,i,o){var c=i.pendingProps,h=jt.current,x=!1,w=(i.flags&128)!==0,N;if((N=w)||(N=n!==null&&n.memoizedState===null?!1:(h&2)!==0),N?(x=!0,i.flags&=-129):(n===null||n.memoizedState!==null)&&(h|=1),kt(jt,h&1),n===null)return Bu(i),n=i.memoizedState,n!==null&&(n=n.dehydrated,n!==null)?((i.mode&1)===0?i.lanes=1:n.data==="$!"?i.lanes=8:i.lanes=1073741824,null):(w=c.children,n=c.fallback,x?(c=i.mode,x=i.child,w={mode:"hidden",children:w},(c&1)===0&&x!==null?(x.childLanes=0,x.pendingProps=w):x=rl(w,c,0,null),n=qr(n,c,o,null),x.return=i,n.return=i,x.sibling=n,i.child=x,i.child.memoizedState=fc(o),i.memoizedState=cc,n):dc(i,w));if(h=n.memoizedState,h!==null&&(N=h.dehydrated,N!==null))return e0(n,i,w,c,N,h,o);if(x){x=c.fallback,w=i.mode,h=n.child,N=h.sibling;var B={mode:"hidden",children:c.children};return(w&1)===0&&i.child!==h?(c=i.child,c.childLanes=0,c.pendingProps=B,i.deletions=null):(c=Mr(h,B),c.subtreeFlags=h.subtreeFlags&14680064),N!==null?x=Mr(N,x):(x=qr(x,w,o,null),x.flags|=2),x.return=i,c.return=i,c.sibling=x,i.child=c,c=x,x=i.child,w=n.child.memoizedState,w=w===null?fc(o):{baseLanes:w.baseLanes|o,cachePool:null,transitions:w.transitions},x.memoizedState=w,x.childLanes=n.childLanes&~o,i.memoizedState=cc,c}return x=n.child,n=x.sibling,c=Mr(x,{mode:"visible",children:c.children}),(i.mode&1)===0&&(c.lanes=o),c.return=i,c.sibling=null,n!==null&&(o=i.deletions,o===null?(i.deletions=[n],i.flags|=16):o.push(n)),i.child=c,i.memoizedState=null,c}function dc(n,i){return i=rl({mode:"visible",children:i},n.mode,0,null),i.return=n,n.child=i}function Xa(n,i,o,c){return c!==null&&ku(c),ws(i,n.child,null,o),n=dc(i,i.pendingProps.children),n.flags|=2,i.memoizedState=null,n}function e0(n,i,o,c,h,x,w){if(o)return i.flags&256?(i.flags&=-257,c=oc(Error(t(422))),Xa(n,i,w,c)):i.memoizedState!==null?(i.child=n.child,i.flags|=128,null):(x=c.fallback,h=i.mode,c=rl({mode:"visible",children:c.children},h,0,null),x=qr(x,h,w,null),x.flags|=2,c.return=i,x.return=i,c.sibling=x,i.child=c,(i.mode&1)!==0&&ws(i,n.child,null,w),i.child.memoizedState=fc(w),i.memoizedState=cc,x);if((i.mode&1)===0)return Xa(n,i,w,null);if(h.data==="$!"){if(c=h.nextSibling&&h.nextSibling.dataset,c)var N=c.dgst;return c=N,x=Error(t(419)),c=oc(x,c,void 0),Xa(n,i,w,c)}if(N=(w&n.childLanes)!==0,Un||N){if(c=cn,c!==null){switch(w&-w){case 4:h=2;break;case 16:h=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:h=32;break;case 536870912:h=268435456;break;default:h=0}h=(h&(c.suspendedLanes|w))!==0?0:h,h!==0&&h!==x.retryLane&&(x.retryLane=h,Vi(n,h),di(c,n,h,-1))}return Cc(),c=oc(Error(t(421))),Xa(n,i,w,c)}return h.data==="$?"?(i.flags|=128,i.child=n.child,i=h0.bind(null,n),h._reactRetry=i,null):(n=x.treeContext,Gn=fr(h.nextSibling),Vn=i,Xt=!0,li=null,n!==null&&(jn[Kn++]=zi,jn[Kn++]=Hi,jn[Kn++]=Br,zi=n.id,Hi=n.overflow,Br=i),i=dc(i,c.children),i.flags|=4096,i)}function gp(n,i,o){n.lanes|=i;var c=n.alternate;c!==null&&(c.lanes|=i),Gu(n.return,i,o)}function hc(n,i,o,c,h){var x=n.memoizedState;x===null?n.memoizedState={isBackwards:i,rendering:null,renderingStartTime:0,last:c,tail:o,tailMode:h}:(x.isBackwards=i,x.rendering=null,x.renderingStartTime=0,x.last=c,x.tail=o,x.tailMode=h)}function vp(n,i,o){var c=i.pendingProps,h=c.revealOrder,x=c.tail;if(An(n,i,c.children,o),c=jt.current,(c&2)!==0)c=c&1|2,i.flags|=128;else{if(n!==null&&(n.flags&128)!==0)e:for(n=i.child;n!==null;){if(n.tag===13)n.memoizedState!==null&&gp(n,o,i);else if(n.tag===19)gp(n,o,i);else if(n.child!==null){n.child.return=n,n=n.child;continue}if(n===i)break e;for(;n.sibling===null;){if(n.return===null||n.return===i)break e;n=n.return}n.sibling.return=n.return,n=n.sibling}c&=1}if(kt(jt,c),(i.mode&1)===0)i.memoizedState=null;else switch(h){case"forwards":for(o=i.child,h=null;o!==null;)n=o.alternate,n!==null&&Ba(n)===null&&(h=o),o=o.sibling;o=h,o===null?(h=i.child,i.child=null):(h=o.sibling,o.sibling=null),hc(i,!1,h,o,x);break;case"backwards":for(o=null,h=i.child,i.child=null;h!==null;){if(n=h.alternate,n!==null&&Ba(n)===null){i.child=h;break}n=h.sibling,h.sibling=o,o=h,h=n}hc(i,!0,o,null,x);break;case"together":hc(i,!1,null,null,void 0);break;default:i.memoizedState=null}return i.child}function Ya(n,i){(i.mode&1)===0&&n!==null&&(n.alternate=null,i.alternate=null,i.flags|=2)}function Wi(n,i,o){if(n!==null&&(i.dependencies=n.dependencies),Gr|=i.lanes,(o&i.childLanes)===0)return null;if(n!==null&&i.child!==n.child)throw Error(t(153));if(i.child!==null){for(n=i.child,o=Mr(n,n.pendingProps),i.child=o,o.return=i;n.sibling!==null;)n=n.sibling,o=o.sibling=Mr(n,n.pendingProps),o.return=i;o.sibling=null}return i.child}function t0(n,i,o){switch(i.tag){case 3:pp(i),Ts();break;case 5:Dh(i);break;case 1:Nn(i.type)&&Ca(i);break;case 4:Yu(i,i.stateNode.containerInfo);break;case 10:var c=i.type._context,h=i.memoizedProps.value;kt(Na,c._currentValue),c._currentValue=h;break;case 13:if(c=i.memoizedState,c!==null)return c.dehydrated!==null?(kt(jt,jt.current&1),i.flags|=128,null):(o&i.child.childLanes)!==0?_p(n,i,o):(kt(jt,jt.current&1),n=Wi(n,i,o),n!==null?n.sibling:null);kt(jt,jt.current&1);break;case 19:if(c=(o&i.childLanes)!==0,(n.flags&128)!==0){if(c)return vp(n,i,o);i.flags|=128}if(h=i.memoizedState,h!==null&&(h.rendering=null,h.tail=null,h.lastEffect=null),kt(jt,jt.current),c)break;return null;case 22:case 23:return i.lanes=0,fp(n,i,o)}return Wi(n,i,o)}var xp,pc,Sp,yp;xp=function(n,i){for(var o=i.child;o!==null;){if(o.tag===5||o.tag===6)n.appendChild(o.stateNode);else if(o.tag!==4&&o.child!==null){o.child.return=o,o=o.child;continue}if(o===i)break;for(;o.sibling===null;){if(o.return===null||o.return===i)return;o=o.return}o.sibling.return=o.return,o=o.sibling}},pc=function(){},Sp=function(n,i,o,c){var h=n.memoizedProps;if(h!==c){n=i.stateNode,Hr(Ei.current);var x=null;switch(o){case"input":h=Et(n,h),c=Et(n,c),x=[];break;case"select":h=ie({},h,{value:void 0}),c=ie({},c,{value:void 0}),x=[];break;case"textarea":h=dt(n,h),c=dt(n,c),x=[];break;default:typeof h.onClick!="function"&&typeof c.onClick=="function"&&(n.onclick=wa)}Be(o,c);var w;o=null;for(ue in h)if(!c.hasOwnProperty(ue)&&h.hasOwnProperty(ue)&&h[ue]!=null)if(ue==="style"){var N=h[ue];for(w in N)N.hasOwnProperty(w)&&(o||(o={}),o[w]="")}else ue!=="dangerouslySetInnerHTML"&&ue!=="children"&&ue!=="suppressContentEditableWarning"&&ue!=="suppressHydrationWarning"&&ue!=="autoFocus"&&(a.hasOwnProperty(ue)?x||(x=[]):(x=x||[]).push(ue,null));for(ue in c){var B=c[ue];if(N=h!=null?h[ue]:void 0,c.hasOwnProperty(ue)&&B!==N&&(B!=null||N!=null))if(ue==="style")if(N){for(w in N)!N.hasOwnProperty(w)||B&&B.hasOwnProperty(w)||(o||(o={}),o[w]="");for(w in B)B.hasOwnProperty(w)&&N[w]!==B[w]&&(o||(o={}),o[w]=B[w])}else o||(x||(x=[]),x.push(ue,o)),o=B;else ue==="dangerouslySetInnerHTML"?(B=B?B.__html:void 0,N=N?N.__html:void 0,B!=null&&N!==B&&(x=x||[]).push(ue,B)):ue==="children"?typeof B!="string"&&typeof B!="number"||(x=x||[]).push(ue,""+B):ue!=="suppressContentEditableWarning"&&ue!=="suppressHydrationWarning"&&(a.hasOwnProperty(ue)?(B!=null&&ue==="onScroll"&&Vt("scroll",n),x||N===B||(x=[])):(x=x||[]).push(ue,B))}o&&(x=x||[]).push("style",o);var ue=x;(i.updateQueue=ue)&&(i.flags|=4)}},yp=function(n,i,o,c){o!==c&&(i.flags|=4)};function Fo(n,i){if(!Xt)switch(n.tailMode){case"hidden":i=n.tail;for(var o=null;i!==null;)i.alternate!==null&&(o=i),i=i.sibling;o===null?n.tail=null:o.sibling=null;break;case"collapsed":o=n.tail;for(var c=null;o!==null;)o.alternate!==null&&(c=o),o=o.sibling;c===null?i||n.tail===null?n.tail=null:n.tail.sibling=null:c.sibling=null}}function Sn(n){var i=n.alternate!==null&&n.alternate.child===n.child,o=0,c=0;if(i)for(var h=n.child;h!==null;)o|=h.lanes|h.childLanes,c|=h.subtreeFlags&14680064,c|=h.flags&14680064,h.return=n,h=h.sibling;else for(h=n.child;h!==null;)o|=h.lanes|h.childLanes,c|=h.subtreeFlags,c|=h.flags,h.return=n,h=h.sibling;return n.subtreeFlags|=c,n.childLanes=o,i}function n0(n,i,o){var c=i.pendingProps;switch(Fu(i),i.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Sn(i),null;case 1:return Nn(i.type)&&Ra(),Sn(i),null;case 3:return c=i.stateNode,Cs(),Gt(In),Gt(vn),Ku(),c.pendingContext&&(c.context=c.pendingContext,c.pendingContext=null),(n===null||n.child===null)&&(Da(i)?i.flags|=4:n===null||n.memoizedState.isDehydrated&&(i.flags&256)===0||(i.flags|=1024,li!==null&&(wc(li),li=null))),pc(n,i),Sn(i),null;case 5:qu(i);var h=Hr(Lo.current);if(o=i.type,n!==null&&i.stateNode!=null)Sp(n,i,o,c,h),n.ref!==i.ref&&(i.flags|=512,i.flags|=2097152);else{if(!c){if(i.stateNode===null)throw Error(t(166));return Sn(i),null}if(n=Hr(Ei.current),Da(i)){c=i.stateNode,o=i.type;var x=i.memoizedProps;switch(c[Mi]=i,c[Ao]=x,n=(i.mode&1)!==0,o){case"dialog":Vt("cancel",c),Vt("close",c);break;case"iframe":case"object":case"embed":Vt("load",c);break;case"video":case"audio":for(h=0;h<Eo.length;h++)Vt(Eo[h],c);break;case"source":Vt("error",c);break;case"img":case"image":case"link":Vt("error",c),Vt("load",c);break;case"details":Vt("toggle",c);break;case"input":Dt(c,x),Vt("invalid",c);break;case"select":c._wrapperState={wasMultiple:!!x.multiple},Vt("invalid",c);break;case"textarea":Ct(c,x),Vt("invalid",c)}Be(o,x),h=null;for(var w in x)if(x.hasOwnProperty(w)){var N=x[w];w==="children"?typeof N=="string"?c.textContent!==N&&(x.suppressHydrationWarning!==!0&&Ta(c.textContent,N,n),h=["children",N]):typeof N=="number"&&c.textContent!==""+N&&(x.suppressHydrationWarning!==!0&&Ta(c.textContent,N,n),h=["children",""+N]):a.hasOwnProperty(w)&&N!=null&&w==="onScroll"&&Vt("scroll",c)}switch(o){case"input":$e(c),Ft(c,x,!0);break;case"textarea":$e(c),zt(c);break;case"select":case"option":break;default:typeof x.onClick=="function"&&(c.onclick=wa)}c=h,i.updateQueue=c,c!==null&&(i.flags|=4)}else{w=h.nodeType===9?h:h.ownerDocument,n==="http://www.w3.org/1999/xhtml"&&(n=b(o)),n==="http://www.w3.org/1999/xhtml"?o==="script"?(n=w.createElement("div"),n.innerHTML="<script><\/script>",n=n.removeChild(n.firstChild)):typeof c.is=="string"?n=w.createElement(o,{is:c.is}):(n=w.createElement(o),o==="select"&&(w=n,c.multiple?w.multiple=!0:c.size&&(w.size=c.size))):n=w.createElementNS(n,o),n[Mi]=i,n[Ao]=c,xp(n,i,!1,!1),i.stateNode=n;e:{switch(w=Ae(o,c),o){case"dialog":Vt("cancel",n),Vt("close",n),h=c;break;case"iframe":case"object":case"embed":Vt("load",n),h=c;break;case"video":case"audio":for(h=0;h<Eo.length;h++)Vt(Eo[h],n);h=c;break;case"source":Vt("error",n),h=c;break;case"img":case"image":case"link":Vt("error",n),Vt("load",n),h=c;break;case"details":Vt("toggle",n),h=c;break;case"input":Dt(n,c),h=Et(n,c),Vt("invalid",n);break;case"option":h=c;break;case"select":n._wrapperState={wasMultiple:!!c.multiple},h=ie({},c,{value:void 0}),Vt("invalid",n);break;case"textarea":Ct(n,c),h=dt(n,c),Vt("invalid",n);break;default:h=c}Be(o,h),N=h;for(x in N)if(N.hasOwnProperty(x)){var B=N[x];x==="style"?pe(n,B):x==="dangerouslySetInnerHTML"?(B=B?B.__html:void 0,B!=null&&he(n,B)):x==="children"?typeof B=="string"?(o!=="textarea"||B!=="")&&me(n,B):typeof B=="number"&&me(n,""+B):x!=="suppressContentEditableWarning"&&x!=="suppressHydrationWarning"&&x!=="autoFocus"&&(a.hasOwnProperty(x)?B!=null&&x==="onScroll"&&Vt("scroll",n):B!=null&&P(n,x,B,w))}switch(o){case"input":$e(n),Ft(n,c,!1);break;case"textarea":$e(n),zt(n);break;case"option":c.value!=null&&n.setAttribute("value",""+de(c.value));break;case"select":n.multiple=!!c.multiple,x=c.value,x!=null?Ot(n,!!c.multiple,x,!1):c.defaultValue!=null&&Ot(n,!!c.multiple,c.defaultValue,!0);break;default:typeof h.onClick=="function"&&(n.onclick=wa)}switch(o){case"button":case"input":case"select":case"textarea":c=!!c.autoFocus;break e;case"img":c=!0;break e;default:c=!1}}c&&(i.flags|=4)}i.ref!==null&&(i.flags|=512,i.flags|=2097152)}return Sn(i),null;case 6:if(n&&i.stateNode!=null)yp(n,i,n.memoizedProps,c);else{if(typeof c!="string"&&i.stateNode===null)throw Error(t(166));if(o=Hr(Lo.current),Hr(Ei.current),Da(i)){if(c=i.stateNode,o=i.memoizedProps,c[Mi]=i,(x=c.nodeValue!==o)&&(n=Vn,n!==null))switch(n.tag){case 3:Ta(c.nodeValue,o,(n.mode&1)!==0);break;case 5:n.memoizedProps.suppressHydrationWarning!==!0&&Ta(c.nodeValue,o,(n.mode&1)!==0)}x&&(i.flags|=4)}else c=(o.nodeType===9?o:o.ownerDocument).createTextNode(c),c[Mi]=i,i.stateNode=c}return Sn(i),null;case 13:if(Gt(jt),c=i.memoizedState,n===null||n.memoizedState!==null&&n.memoizedState.dehydrated!==null){if(Xt&&Gn!==null&&(i.mode&1)!==0&&(i.flags&128)===0)Th(),Ts(),i.flags|=98560,x=!1;else if(x=Da(i),c!==null&&c.dehydrated!==null){if(n===null){if(!x)throw Error(t(318));if(x=i.memoizedState,x=x!==null?x.dehydrated:null,!x)throw Error(t(317));x[Mi]=i}else Ts(),(i.flags&128)===0&&(i.memoizedState=null),i.flags|=4;Sn(i),x=!1}else li!==null&&(wc(li),li=null),x=!0;if(!x)return i.flags&65536?i:null}return(i.flags&128)!==0?(i.lanes=o,i):(c=c!==null,c!==(n!==null&&n.memoizedState!==null)&&c&&(i.child.flags|=8192,(i.mode&1)!==0&&(n===null||(jt.current&1)!==0?on===0&&(on=3):Cc())),i.updateQueue!==null&&(i.flags|=4),Sn(i),null);case 4:return Cs(),pc(n,i),n===null&&To(i.stateNode.containerInfo),Sn(i),null;case 10:return Vu(i.type._context),Sn(i),null;case 17:return Nn(i.type)&&Ra(),Sn(i),null;case 19:if(Gt(jt),x=i.memoizedState,x===null)return Sn(i),null;if(c=(i.flags&128)!==0,w=x.rendering,w===null)if(c)Fo(x,!1);else{if(on!==0||n!==null&&(n.flags&128)!==0)for(n=i.child;n!==null;){if(w=Ba(n),w!==null){for(i.flags|=128,Fo(x,!1),c=w.updateQueue,c!==null&&(i.updateQueue=c,i.flags|=4),i.subtreeFlags=0,c=o,o=i.child;o!==null;)x=o,n=c,x.flags&=14680066,w=x.alternate,w===null?(x.childLanes=0,x.lanes=n,x.child=null,x.subtreeFlags=0,x.memoizedProps=null,x.memoizedState=null,x.updateQueue=null,x.dependencies=null,x.stateNode=null):(x.childLanes=w.childLanes,x.lanes=w.lanes,x.child=w.child,x.subtreeFlags=0,x.deletions=null,x.memoizedProps=w.memoizedProps,x.memoizedState=w.memoizedState,x.updateQueue=w.updateQueue,x.type=w.type,n=w.dependencies,x.dependencies=n===null?null:{lanes:n.lanes,firstContext:n.firstContext}),o=o.sibling;return kt(jt,jt.current&1|2),i.child}n=n.sibling}x.tail!==null&&qt()>Ds&&(i.flags|=128,c=!0,Fo(x,!1),i.lanes=4194304)}else{if(!c)if(n=Ba(w),n!==null){if(i.flags|=128,c=!0,o=n.updateQueue,o!==null&&(i.updateQueue=o,i.flags|=4),Fo(x,!0),x.tail===null&&x.tailMode==="hidden"&&!w.alternate&&!Xt)return Sn(i),null}else 2*qt()-x.renderingStartTime>Ds&&o!==1073741824&&(i.flags|=128,c=!0,Fo(x,!1),i.lanes=4194304);x.isBackwards?(w.sibling=i.child,i.child=w):(o=x.last,o!==null?o.sibling=w:i.child=w,x.last=w)}return x.tail!==null?(i=x.tail,x.rendering=i,x.tail=i.sibling,x.renderingStartTime=qt(),i.sibling=null,o=jt.current,kt(jt,c?o&1|2:o&1),i):(Sn(i),null);case 22:case 23:return Rc(),c=i.memoizedState!==null,n!==null&&n.memoizedState!==null!==c&&(i.flags|=8192),c&&(i.mode&1)!==0?(Wn&1073741824)!==0&&(Sn(i),i.subtreeFlags&6&&(i.flags|=8192)):Sn(i),null;case 24:return null;case 25:return null}throw Error(t(156,i.tag))}function i0(n,i){switch(Fu(i),i.tag){case 1:return Nn(i.type)&&Ra(),n=i.flags,n&65536?(i.flags=n&-65537|128,i):null;case 3:return Cs(),Gt(In),Gt(vn),Ku(),n=i.flags,(n&65536)!==0&&(n&128)===0?(i.flags=n&-65537|128,i):null;case 5:return qu(i),null;case 13:if(Gt(jt),n=i.memoizedState,n!==null&&n.dehydrated!==null){if(i.alternate===null)throw Error(t(340));Ts()}return n=i.flags,n&65536?(i.flags=n&-65537|128,i):null;case 19:return Gt(jt),null;case 4:return Cs(),null;case 10:return Vu(i.type._context),null;case 22:case 23:return Rc(),null;case 24:return null;default:return null}}var qa=!1,yn=!1,r0=typeof WeakSet=="function"?WeakSet:Set,Ve=null;function Ps(n,i){var o=n.ref;if(o!==null)if(typeof o=="function")try{o(null)}catch(c){$t(n,i,c)}else o.current=null}function mc(n,i,o){try{o()}catch(c){$t(n,i,c)}}var Mp=!1;function s0(n,i){if(Ru=ha,n=eh(),xu(n)){if("selectionStart"in n)var o={start:n.selectionStart,end:n.selectionEnd};else e:{o=(o=n.ownerDocument)&&o.defaultView||window;var c=o.getSelection&&o.getSelection();if(c&&c.rangeCount!==0){o=c.anchorNode;var h=c.anchorOffset,x=c.focusNode;c=c.focusOffset;try{o.nodeType,x.nodeType}catch{o=null;break e}var w=0,N=-1,B=-1,ue=0,xe=0,Se=n,ve=null;t:for(;;){for(var ze;Se!==o||h!==0&&Se.nodeType!==3||(N=w+h),Se!==x||c!==0&&Se.nodeType!==3||(B=w+c),Se.nodeType===3&&(w+=Se.nodeValue.length),(ze=Se.firstChild)!==null;)ve=Se,Se=ze;for(;;){if(Se===n)break t;if(ve===o&&++ue===h&&(N=w),ve===x&&++xe===c&&(B=w),(ze=Se.nextSibling)!==null)break;Se=ve,ve=Se.parentNode}Se=ze}o=N===-1||B===-1?null:{start:N,end:B}}else o=null}o=o||{start:0,end:0}}else o=null;for(Cu={focusedElem:n,selectionRange:o},ha=!1,Ve=i;Ve!==null;)if(i=Ve,n=i.child,(i.subtreeFlags&1028)!==0&&n!==null)n.return=i,Ve=n;else for(;Ve!==null;){i=Ve;try{var Ge=i.alternate;if((i.flags&1024)!==0)switch(i.tag){case 0:case 11:case 15:break;case 1:if(Ge!==null){var Ye=Ge.memoizedProps,Qt=Ge.memoizedState,Q=i.stateNode,V=Q.getSnapshotBeforeUpdate(i.elementType===i.type?Ye:ui(i.type,Ye),Qt);Q.__reactInternalSnapshotBeforeUpdate=V}break;case 3:var ne=i.stateNode.containerInfo;ne.nodeType===1?ne.textContent="":ne.nodeType===9&&ne.documentElement&&ne.removeChild(ne.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(t(163))}}catch(Ee){$t(i,i.return,Ee)}if(n=i.sibling,n!==null){n.return=i.return,Ve=n;break}Ve=i.return}return Ge=Mp,Mp=!1,Ge}function Oo(n,i,o){var c=i.updateQueue;if(c=c!==null?c.lastEffect:null,c!==null){var h=c=c.next;do{if((h.tag&n)===n){var x=h.destroy;h.destroy=void 0,x!==void 0&&mc(i,o,x)}h=h.next}while(h!==c)}}function ja(n,i){if(i=i.updateQueue,i=i!==null?i.lastEffect:null,i!==null){var o=i=i.next;do{if((o.tag&n)===n){var c=o.create;o.destroy=c()}o=o.next}while(o!==i)}}function _c(n){var i=n.ref;if(i!==null){var o=n.stateNode;switch(n.tag){case 5:n=o;break;default:n=o}typeof i=="function"?i(n):i.current=n}}function Ep(n){var i=n.alternate;i!==null&&(n.alternate=null,Ep(i)),n.child=null,n.deletions=null,n.sibling=null,n.tag===5&&(i=n.stateNode,i!==null&&(delete i[Mi],delete i[Ao],delete i[Du],delete i[Hg],delete i[Vg])),n.stateNode=null,n.return=null,n.dependencies=null,n.memoizedProps=null,n.memoizedState=null,n.pendingProps=null,n.stateNode=null,n.updateQueue=null}function Tp(n){return n.tag===5||n.tag===3||n.tag===4}function wp(n){e:for(;;){for(;n.sibling===null;){if(n.return===null||Tp(n.return))return null;n=n.return}for(n.sibling.return=n.return,n=n.sibling;n.tag!==5&&n.tag!==6&&n.tag!==18;){if(n.flags&2||n.child===null||n.tag===4)continue e;n.child.return=n,n=n.child}if(!(n.flags&2))return n.stateNode}}function gc(n,i,o){var c=n.tag;if(c===5||c===6)n=n.stateNode,i?o.nodeType===8?o.parentNode.insertBefore(n,i):o.insertBefore(n,i):(o.nodeType===8?(i=o.parentNode,i.insertBefore(n,o)):(i=o,i.appendChild(n)),o=o._reactRootContainer,o!=null||i.onclick!==null||(i.onclick=wa));else if(c!==4&&(n=n.child,n!==null))for(gc(n,i,o),n=n.sibling;n!==null;)gc(n,i,o),n=n.sibling}function vc(n,i,o){var c=n.tag;if(c===5||c===6)n=n.stateNode,i?o.insertBefore(n,i):o.appendChild(n);else if(c!==4&&(n=n.child,n!==null))for(vc(n,i,o),n=n.sibling;n!==null;)vc(n,i,o),n=n.sibling}var mn=null,ci=!1;function gr(n,i,o){for(o=o.child;o!==null;)Ap(n,i,o),o=o.sibling}function Ap(n,i,o){if(be&&typeof be.onCommitFiberUnmount=="function")try{be.onCommitFiberUnmount(ee,o)}catch{}switch(o.tag){case 5:yn||Ps(o,i);case 6:var c=mn,h=ci;mn=null,gr(n,i,o),mn=c,ci=h,mn!==null&&(ci?(n=mn,o=o.stateNode,n.nodeType===8?n.parentNode.removeChild(o):n.removeChild(o)):mn.removeChild(o.stateNode));break;case 18:mn!==null&&(ci?(n=mn,o=o.stateNode,n.nodeType===8?Lu(n.parentNode,o):n.nodeType===1&&Lu(n,o),mo(n)):Lu(mn,o.stateNode));break;case 4:c=mn,h=ci,mn=o.stateNode.containerInfo,ci=!0,gr(n,i,o),mn=c,ci=h;break;case 0:case 11:case 14:case 15:if(!yn&&(c=o.updateQueue,c!==null&&(c=c.lastEffect,c!==null))){h=c=c.next;do{var x=h,w=x.destroy;x=x.tag,w!==void 0&&((x&2)!==0||(x&4)!==0)&&mc(o,i,w),h=h.next}while(h!==c)}gr(n,i,o);break;case 1:if(!yn&&(Ps(o,i),c=o.stateNode,typeof c.componentWillUnmount=="function"))try{c.props=o.memoizedProps,c.state=o.memoizedState,c.componentWillUnmount()}catch(N){$t(o,i,N)}gr(n,i,o);break;case 21:gr(n,i,o);break;case 22:o.mode&1?(yn=(c=yn)||o.memoizedState!==null,gr(n,i,o),yn=c):gr(n,i,o);break;default:gr(n,i,o)}}function Rp(n){var i=n.updateQueue;if(i!==null){n.updateQueue=null;var o=n.stateNode;o===null&&(o=n.stateNode=new r0),i.forEach(function(c){var h=p0.bind(null,n,c);o.has(c)||(o.add(c),c.then(h,h))})}}function fi(n,i){var o=i.deletions;if(o!==null)for(var c=0;c<o.length;c++){var h=o[c];try{var x=n,w=i,N=w;e:for(;N!==null;){switch(N.tag){case 5:mn=N.stateNode,ci=!1;break e;case 3:mn=N.stateNode.containerInfo,ci=!0;break e;case 4:mn=N.stateNode.containerInfo,ci=!0;break e}N=N.return}if(mn===null)throw Error(t(160));Ap(x,w,h),mn=null,ci=!1;var B=h.alternate;B!==null&&(B.return=null),h.return=null}catch(ue){$t(h,i,ue)}}if(i.subtreeFlags&12854)for(i=i.child;i!==null;)Cp(i,n),i=i.sibling}function Cp(n,i){var o=n.alternate,c=n.flags;switch(n.tag){case 0:case 11:case 14:case 15:if(fi(i,n),wi(n),c&4){try{Oo(3,n,n.return),ja(3,n)}catch(Ye){$t(n,n.return,Ye)}try{Oo(5,n,n.return)}catch(Ye){$t(n,n.return,Ye)}}break;case 1:fi(i,n),wi(n),c&512&&o!==null&&Ps(o,o.return);break;case 5:if(fi(i,n),wi(n),c&512&&o!==null&&Ps(o,o.return),n.flags&32){var h=n.stateNode;try{me(h,"")}catch(Ye){$t(n,n.return,Ye)}}if(c&4&&(h=n.stateNode,h!=null)){var x=n.memoizedProps,w=o!==null?o.memoizedProps:x,N=n.type,B=n.updateQueue;if(n.updateQueue=null,B!==null)try{N==="input"&&x.type==="radio"&&x.name!=null&&ft(h,x),Ae(N,w);var ue=Ae(N,x);for(w=0;w<B.length;w+=2){var xe=B[w],Se=B[w+1];xe==="style"?pe(h,Se):xe==="dangerouslySetInnerHTML"?he(h,Se):xe==="children"?me(h,Se):P(h,xe,Se,ue)}switch(N){case"input":Yt(h,x);break;case"textarea":Ne(h,x);break;case"select":var ve=h._wrapperState.wasMultiple;h._wrapperState.wasMultiple=!!x.multiple;var ze=x.value;ze!=null?Ot(h,!!x.multiple,ze,!1):ve!==!!x.multiple&&(x.defaultValue!=null?Ot(h,!!x.multiple,x.defaultValue,!0):Ot(h,!!x.multiple,x.multiple?[]:"",!1))}h[Ao]=x}catch(Ye){$t(n,n.return,Ye)}}break;case 6:if(fi(i,n),wi(n),c&4){if(n.stateNode===null)throw Error(t(162));h=n.stateNode,x=n.memoizedProps;try{h.nodeValue=x}catch(Ye){$t(n,n.return,Ye)}}break;case 3:if(fi(i,n),wi(n),c&4&&o!==null&&o.memoizedState.isDehydrated)try{mo(i.containerInfo)}catch(Ye){$t(n,n.return,Ye)}break;case 4:fi(i,n),wi(n);break;case 13:fi(i,n),wi(n),h=n.child,h.flags&8192&&(x=h.memoizedState!==null,h.stateNode.isHidden=x,!x||h.alternate!==null&&h.alternate.memoizedState!==null||(yc=qt())),c&4&&Rp(n);break;case 22:if(xe=o!==null&&o.memoizedState!==null,n.mode&1?(yn=(ue=yn)||xe,fi(i,n),yn=ue):fi(i,n),wi(n),c&8192){if(ue=n.memoizedState!==null,(n.stateNode.isHidden=ue)&&!xe&&(n.mode&1)!==0)for(Ve=n,xe=n.child;xe!==null;){for(Se=Ve=xe;Ve!==null;){switch(ve=Ve,ze=ve.child,ve.tag){case 0:case 11:case 14:case 15:Oo(4,ve,ve.return);break;case 1:Ps(ve,ve.return);var Ge=ve.stateNode;if(typeof Ge.componentWillUnmount=="function"){c=ve,o=ve.return;try{i=c,Ge.props=i.memoizedProps,Ge.state=i.memoizedState,Ge.componentWillUnmount()}catch(Ye){$t(c,o,Ye)}}break;case 5:Ps(ve,ve.return);break;case 22:if(ve.memoizedState!==null){Lp(Se);continue}}ze!==null?(ze.return=ve,Ve=ze):Lp(Se)}xe=xe.sibling}e:for(xe=null,Se=n;;){if(Se.tag===5){if(xe===null){xe=Se;try{h=Se.stateNode,ue?(x=h.style,typeof x.setProperty=="function"?x.setProperty("display","none","important"):x.display="none"):(N=Se.stateNode,B=Se.memoizedProps.style,w=B!=null&&B.hasOwnProperty("display")?B.display:null,N.style.display=ce("display",w))}catch(Ye){$t(n,n.return,Ye)}}}else if(Se.tag===6){if(xe===null)try{Se.stateNode.nodeValue=ue?"":Se.memoizedProps}catch(Ye){$t(n,n.return,Ye)}}else if((Se.tag!==22&&Se.tag!==23||Se.memoizedState===null||Se===n)&&Se.child!==null){Se.child.return=Se,Se=Se.child;continue}if(Se===n)break e;for(;Se.sibling===null;){if(Se.return===null||Se.return===n)break e;xe===Se&&(xe=null),Se=Se.return}xe===Se&&(xe=null),Se.sibling.return=Se.return,Se=Se.sibling}}break;case 19:fi(i,n),wi(n),c&4&&Rp(n);break;case 21:break;default:fi(i,n),wi(n)}}function wi(n){var i=n.flags;if(i&2){try{e:{for(var o=n.return;o!==null;){if(Tp(o)){var c=o;break e}o=o.return}throw Error(t(160))}switch(c.tag){case 5:var h=c.stateNode;c.flags&32&&(me(h,""),c.flags&=-33);var x=wp(n);vc(n,x,h);break;case 3:case 4:var w=c.stateNode.containerInfo,N=wp(n);gc(n,N,w);break;default:throw Error(t(161))}}catch(B){$t(n,n.return,B)}n.flags&=-3}i&4096&&(n.flags&=-4097)}function o0(n,i,o){Ve=n,bp(n)}function bp(n,i,o){for(var c=(n.mode&1)!==0;Ve!==null;){var h=Ve,x=h.child;if(h.tag===22&&c){var w=h.memoizedState!==null||qa;if(!w){var N=h.alternate,B=N!==null&&N.memoizedState!==null||yn;N=qa;var ue=yn;if(qa=w,(yn=B)&&!ue)for(Ve=h;Ve!==null;)w=Ve,B=w.child,w.tag===22&&w.memoizedState!==null?Dp(h):B!==null?(B.return=w,Ve=B):Dp(h);for(;x!==null;)Ve=x,bp(x),x=x.sibling;Ve=h,qa=N,yn=ue}Pp(n)}else(h.subtreeFlags&8772)!==0&&x!==null?(x.return=h,Ve=x):Pp(n)}}function Pp(n){for(;Ve!==null;){var i=Ve;if((i.flags&8772)!==0){var o=i.alternate;try{if((i.flags&8772)!==0)switch(i.tag){case 0:case 11:case 15:yn||ja(5,i);break;case 1:var c=i.stateNode;if(i.flags&4&&!yn)if(o===null)c.componentDidMount();else{var h=i.elementType===i.type?o.memoizedProps:ui(i.type,o.memoizedProps);c.componentDidUpdate(h,o.memoizedState,c.__reactInternalSnapshotBeforeUpdate)}var x=i.updateQueue;x!==null&&Lh(i,x,c);break;case 3:var w=i.updateQueue;if(w!==null){if(o=null,i.child!==null)switch(i.child.tag){case 5:o=i.child.stateNode;break;case 1:o=i.child.stateNode}Lh(i,w,o)}break;case 5:var N=i.stateNode;if(o===null&&i.flags&4){o=N;var B=i.memoizedProps;switch(i.type){case"button":case"input":case"select":case"textarea":B.autoFocus&&o.focus();break;case"img":B.src&&(o.src=B.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(i.memoizedState===null){var ue=i.alternate;if(ue!==null){var xe=ue.memoizedState;if(xe!==null){var Se=xe.dehydrated;Se!==null&&mo(Se)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(t(163))}yn||i.flags&512&&_c(i)}catch(ve){$t(i,i.return,ve)}}if(i===n){Ve=null;break}if(o=i.sibling,o!==null){o.return=i.return,Ve=o;break}Ve=i.return}}function Lp(n){for(;Ve!==null;){var i=Ve;if(i===n){Ve=null;break}var o=i.sibling;if(o!==null){o.return=i.return,Ve=o;break}Ve=i.return}}function Dp(n){for(;Ve!==null;){var i=Ve;try{switch(i.tag){case 0:case 11:case 15:var o=i.return;try{ja(4,i)}catch(B){$t(i,o,B)}break;case 1:var c=i.stateNode;if(typeof c.componentDidMount=="function"){var h=i.return;try{c.componentDidMount()}catch(B){$t(i,h,B)}}var x=i.return;try{_c(i)}catch(B){$t(i,x,B)}break;case 5:var w=i.return;try{_c(i)}catch(B){$t(i,w,B)}}}catch(B){$t(i,i.return,B)}if(i===n){Ve=null;break}var N=i.sibling;if(N!==null){N.return=i.return,Ve=N;break}Ve=i.return}}var a0=Math.ceil,Ka=L.ReactCurrentDispatcher,xc=L.ReactCurrentOwner,Qn=L.ReactCurrentBatchConfig,yt=0,cn=null,tn=null,_n=0,Wn=0,Ls=dr(0),on=0,Bo=null,Gr=0,$a=0,Sc=0,ko=null,Fn=null,yc=0,Ds=1/0,Xi=null,Za=!1,Mc=null,vr=null,Qa=!1,xr=null,Ja=0,zo=0,Ec=null,el=-1,tl=0;function Rn(){return(yt&6)!==0?qt():el!==-1?el:el=qt()}function Sr(n){return(n.mode&1)===0?1:(yt&2)!==0&&_n!==0?_n&-_n:Wg.transition!==null?(tl===0&&(tl=ke()),tl):(n=_t,n!==0||(n=window.event,n=n===void 0?16:Nd(n.type)),n)}function di(n,i,o,c){if(50<zo)throw zo=0,Ec=null,Error(t(185));mt(n,o,c),((yt&2)===0||n!==cn)&&(n===cn&&((yt&2)===0&&($a|=o),on===4&&yr(n,_n)),On(n,c),o===1&&yt===0&&(i.mode&1)===0&&(Ds=qt()+500,ba&&pr()))}function On(n,i){var o=n.callbackNode;bt(n,i);var c=Bt(n,n===cn?_n:0);if(c===0)o!==null&&fa(o),n.callbackNode=null,n.callbackPriority=0;else if(i=c&-c,n.callbackPriority!==i){if(o!=null&&fa(o),i===1)n.tag===0?Gg(Np.bind(null,n)):xh(Np.bind(null,n)),kg(function(){(yt&6)===0&&pr()}),o=null;else{switch(Oi(c)){case 1:o=uo;break;case 4:o=C;break;case 16:o=Y;break;case 536870912:o=te;break;default:o=Y}o=Vp(o,Ip.bind(null,n))}n.callbackPriority=i,n.callbackNode=o}}function Ip(n,i){if(el=-1,tl=0,(yt&6)!==0)throw Error(t(327));var o=n.callbackNode;if(Is()&&n.callbackNode!==o)return null;var c=Bt(n,n===cn?_n:0);if(c===0)return null;if((c&30)!==0||(c&n.expiredLanes)!==0||i)i=nl(n,c);else{i=c;var h=yt;yt|=2;var x=Fp();(cn!==n||_n!==i)&&(Xi=null,Ds=qt()+500,Xr(n,i));do try{c0();break}catch(N){Up(n,N)}while(!0);Hu(),Ka.current=x,yt=h,tn!==null?i=0:(cn=null,_n=0,i=on)}if(i!==0){if(i===2&&(h=en(n),h!==0&&(c=h,i=Tc(n,h))),i===1)throw o=Bo,Xr(n,0),yr(n,c),On(n,qt()),o;if(i===6)yr(n,c);else{if(h=n.current.alternate,(c&30)===0&&!l0(h)&&(i=nl(n,c),i===2&&(x=en(n),x!==0&&(c=x,i=Tc(n,x))),i===1))throw o=Bo,Xr(n,0),yr(n,c),On(n,qt()),o;switch(n.finishedWork=h,n.finishedLanes=c,i){case 0:case 1:throw Error(t(345));case 2:Yr(n,Fn,Xi);break;case 3:if(yr(n,c),(c&130023424)===c&&(i=yc+500-qt(),10<i)){if(Bt(n,0)!==0)break;if(h=n.suspendedLanes,(h&c)!==c){Rn(),n.pingedLanes|=n.suspendedLanes&h;break}n.timeoutHandle=Pu(Yr.bind(null,n,Fn,Xi),i);break}Yr(n,Fn,Xi);break;case 4:if(yr(n,c),(c&4194240)===c)break;for(i=n.eventTimes,h=-1;0<c;){var w=31-Re(c);x=1<<w,w=i[w],w>h&&(h=w),c&=~x}if(c=h,c=qt()-c,c=(120>c?120:480>c?480:1080>c?1080:1920>c?1920:3e3>c?3e3:4320>c?4320:1960*a0(c/1960))-c,10<c){n.timeoutHandle=Pu(Yr.bind(null,n,Fn,Xi),c);break}Yr(n,Fn,Xi);break;case 5:Yr(n,Fn,Xi);break;default:throw Error(t(329))}}}return On(n,qt()),n.callbackNode===o?Ip.bind(null,n):null}function Tc(n,i){var o=ko;return n.current.memoizedState.isDehydrated&&(Xr(n,i).flags|=256),n=nl(n,i),n!==2&&(i=Fn,Fn=o,i!==null&&wc(i)),n}function wc(n){Fn===null?Fn=n:Fn.push.apply(Fn,n)}function l0(n){for(var i=n;;){if(i.flags&16384){var o=i.updateQueue;if(o!==null&&(o=o.stores,o!==null))for(var c=0;c<o.length;c++){var h=o[c],x=h.getSnapshot;h=h.value;try{if(!ai(x(),h))return!1}catch{return!1}}}if(o=i.child,i.subtreeFlags&16384&&o!==null)o.return=i,i=o;else{if(i===n)break;for(;i.sibling===null;){if(i.return===null||i.return===n)return!0;i=i.return}i.sibling.return=i.return,i=i.sibling}}return!0}function yr(n,i){for(i&=~Sc,i&=~$a,n.suspendedLanes|=i,n.pingedLanes&=~i,n=n.expirationTimes;0<i;){var o=31-Re(i),c=1<<o;n[o]=-1,i&=~c}}function Np(n){if((yt&6)!==0)throw Error(t(327));Is();var i=Bt(n,0);if((i&1)===0)return On(n,qt()),null;var o=nl(n,i);if(n.tag!==0&&o===2){var c=en(n);c!==0&&(i=c,o=Tc(n,c))}if(o===1)throw o=Bo,Xr(n,0),yr(n,i),On(n,qt()),o;if(o===6)throw Error(t(345));return n.finishedWork=n.current.alternate,n.finishedLanes=i,Yr(n,Fn,Xi),On(n,qt()),null}function Ac(n,i){var o=yt;yt|=1;try{return n(i)}finally{yt=o,yt===0&&(Ds=qt()+500,ba&&pr())}}function Wr(n){xr!==null&&xr.tag===0&&(yt&6)===0&&Is();var i=yt;yt|=1;var o=Qn.transition,c=_t;try{if(Qn.transition=null,_t=1,n)return n()}finally{_t=c,Qn.transition=o,yt=i,(yt&6)===0&&pr()}}function Rc(){Wn=Ls.current,Gt(Ls)}function Xr(n,i){n.finishedWork=null,n.finishedLanes=0;var o=n.timeoutHandle;if(o!==-1&&(n.timeoutHandle=-1,Bg(o)),tn!==null)for(o=tn.return;o!==null;){var c=o;switch(Fu(c),c.tag){case 1:c=c.type.childContextTypes,c!=null&&Ra();break;case 3:Cs(),Gt(In),Gt(vn),Ku();break;case 5:qu(c);break;case 4:Cs();break;case 13:Gt(jt);break;case 19:Gt(jt);break;case 10:Vu(c.type._context);break;case 22:case 23:Rc()}o=o.return}if(cn=n,tn=n=Mr(n.current,null),_n=Wn=i,on=0,Bo=null,Sc=$a=Gr=0,Fn=ko=null,zr!==null){for(i=0;i<zr.length;i++)if(o=zr[i],c=o.interleaved,c!==null){o.interleaved=null;var h=c.next,x=o.pending;if(x!==null){var w=x.next;x.next=h,c.next=w}o.pending=c}zr=null}return n}function Up(n,i){do{var o=tn;try{if(Hu(),ka.current=Ga,za){for(var c=Kt.memoizedState;c!==null;){var h=c.queue;h!==null&&(h.pending=null),c=c.next}za=!1}if(Vr=0,un=sn=Kt=null,Do=!1,Io=0,xc.current=null,o===null||o.return===null){on=1,Bo=i,tn=null;break}e:{var x=n,w=o.return,N=o,B=i;if(i=_n,N.flags|=32768,B!==null&&typeof B=="object"&&typeof B.then=="function"){var ue=B,xe=N,Se=xe.tag;if((xe.mode&1)===0&&(Se===0||Se===11||Se===15)){var ve=xe.alternate;ve?(xe.updateQueue=ve.updateQueue,xe.memoizedState=ve.memoizedState,xe.lanes=ve.lanes):(xe.updateQueue=null,xe.memoizedState=null)}var ze=op(w);if(ze!==null){ze.flags&=-257,ap(ze,w,N,x,i),ze.mode&1&&sp(x,ue,i),i=ze,B=ue;var Ge=i.updateQueue;if(Ge===null){var Ye=new Set;Ye.add(B),i.updateQueue=Ye}else Ge.add(B);break e}else{if((i&1)===0){sp(x,ue,i),Cc();break e}B=Error(t(426))}}else if(Xt&&N.mode&1){var Qt=op(w);if(Qt!==null){(Qt.flags&65536)===0&&(Qt.flags|=256),ap(Qt,w,N,x,i),ku(bs(B,N));break e}}x=B=bs(B,N),on!==4&&(on=2),ko===null?ko=[x]:ko.push(x),x=w;do{switch(x.tag){case 3:x.flags|=65536,i&=-i,x.lanes|=i;var Q=ip(x,B,i);Ph(x,Q);break e;case 1:N=B;var V=x.type,ne=x.stateNode;if((x.flags&128)===0&&(typeof V.getDerivedStateFromError=="function"||ne!==null&&typeof ne.componentDidCatch=="function"&&(vr===null||!vr.has(ne)))){x.flags|=65536,i&=-i,x.lanes|=i;var Ee=rp(x,N,i);Ph(x,Ee);break e}}x=x.return}while(x!==null)}Bp(o)}catch(Ke){i=Ke,tn===o&&o!==null&&(tn=o=o.return);continue}break}while(!0)}function Fp(){var n=Ka.current;return Ka.current=Ga,n===null?Ga:n}function Cc(){(on===0||on===3||on===2)&&(on=4),cn===null||(Gr&268435455)===0&&($a&268435455)===0||yr(cn,_n)}function nl(n,i){var o=yt;yt|=2;var c=Fp();(cn!==n||_n!==i)&&(Xi=null,Xr(n,i));do try{u0();break}catch(h){Up(n,h)}while(!0);if(Hu(),yt=o,Ka.current=c,tn!==null)throw Error(t(261));return cn=null,_n=0,on}function u0(){for(;tn!==null;)Op(tn)}function c0(){for(;tn!==null&&!su();)Op(tn)}function Op(n){var i=Hp(n.alternate,n,Wn);n.memoizedProps=n.pendingProps,i===null?Bp(n):tn=i,xc.current=null}function Bp(n){var i=n;do{var o=i.alternate;if(n=i.return,(i.flags&32768)===0){if(o=n0(o,i,Wn),o!==null){tn=o;return}}else{if(o=i0(o,i),o!==null){o.flags&=32767,tn=o;return}if(n!==null)n.flags|=32768,n.subtreeFlags=0,n.deletions=null;else{on=6,tn=null;return}}if(i=i.sibling,i!==null){tn=i;return}tn=i=n}while(i!==null);on===0&&(on=5)}function Yr(n,i,o){var c=_t,h=Qn.transition;try{Qn.transition=null,_t=1,f0(n,i,o,c)}finally{Qn.transition=h,_t=c}return null}function f0(n,i,o,c){do Is();while(xr!==null);if((yt&6)!==0)throw Error(t(327));o=n.finishedWork;var h=n.finishedLanes;if(o===null)return null;if(n.finishedWork=null,n.finishedLanes=0,o===n.current)throw Error(t(177));n.callbackNode=null,n.callbackPriority=0;var x=o.lanes|o.childLanes;if(Ln(n,x),n===cn&&(tn=cn=null,_n=0),(o.subtreeFlags&2064)===0&&(o.flags&2064)===0||Qa||(Qa=!0,Vp(Y,function(){return Is(),null})),x=(o.flags&15990)!==0,(o.subtreeFlags&15990)!==0||x){x=Qn.transition,Qn.transition=null;var w=_t;_t=1;var N=yt;yt|=4,xc.current=null,s0(n,o),Cp(o,n),Lg(Cu),ha=!!Ru,Cu=Ru=null,n.current=o,o0(o),ou(),yt=N,_t=w,Qn.transition=x}else n.current=o;if(Qa&&(Qa=!1,xr=n,Ja=h),x=n.pendingLanes,x===0&&(vr=null),He(o.stateNode),On(n,qt()),i!==null)for(c=n.onRecoverableError,o=0;o<i.length;o++)h=i[o],c(h.value,{componentStack:h.stack,digest:h.digest});if(Za)throw Za=!1,n=Mc,Mc=null,n;return(Ja&1)!==0&&n.tag!==0&&Is(),x=n.pendingLanes,(x&1)!==0?n===Ec?zo++:(zo=0,Ec=n):zo=0,pr(),null}function Is(){if(xr!==null){var n=Oi(Ja),i=Qn.transition,o=_t;try{if(Qn.transition=null,_t=16>n?16:n,xr===null)var c=!1;else{if(n=xr,xr=null,Ja=0,(yt&6)!==0)throw Error(t(331));var h=yt;for(yt|=4,Ve=n.current;Ve!==null;){var x=Ve,w=x.child;if((Ve.flags&16)!==0){var N=x.deletions;if(N!==null){for(var B=0;B<N.length;B++){var ue=N[B];for(Ve=ue;Ve!==null;){var xe=Ve;switch(xe.tag){case 0:case 11:case 15:Oo(8,xe,x)}var Se=xe.child;if(Se!==null)Se.return=xe,Ve=Se;else for(;Ve!==null;){xe=Ve;var ve=xe.sibling,ze=xe.return;if(Ep(xe),xe===ue){Ve=null;break}if(ve!==null){ve.return=ze,Ve=ve;break}Ve=ze}}}var Ge=x.alternate;if(Ge!==null){var Ye=Ge.child;if(Ye!==null){Ge.child=null;do{var Qt=Ye.sibling;Ye.sibling=null,Ye=Qt}while(Ye!==null)}}Ve=x}}if((x.subtreeFlags&2064)!==0&&w!==null)w.return=x,Ve=w;else e:for(;Ve!==null;){if(x=Ve,(x.flags&2048)!==0)switch(x.tag){case 0:case 11:case 15:Oo(9,x,x.return)}var Q=x.sibling;if(Q!==null){Q.return=x.return,Ve=Q;break e}Ve=x.return}}var V=n.current;for(Ve=V;Ve!==null;){w=Ve;var ne=w.child;if((w.subtreeFlags&2064)!==0&&ne!==null)ne.return=w,Ve=ne;else e:for(w=V;Ve!==null;){if(N=Ve,(N.flags&2048)!==0)try{switch(N.tag){case 0:case 11:case 15:ja(9,N)}}catch(Ke){$t(N,N.return,Ke)}if(N===w){Ve=null;break e}var Ee=N.sibling;if(Ee!==null){Ee.return=N.return,Ve=Ee;break e}Ve=N.return}}if(yt=h,pr(),be&&typeof be.onPostCommitFiberRoot=="function")try{be.onPostCommitFiberRoot(ee,n)}catch{}c=!0}return c}finally{_t=o,Qn.transition=i}}return!1}function kp(n,i,o){i=bs(o,i),i=ip(n,i,1),n=_r(n,i,1),i=Rn(),n!==null&&(mt(n,1,i),On(n,i))}function $t(n,i,o){if(n.tag===3)kp(n,n,o);else for(;i!==null;){if(i.tag===3){kp(i,n,o);break}else if(i.tag===1){var c=i.stateNode;if(typeof i.type.getDerivedStateFromError=="function"||typeof c.componentDidCatch=="function"&&(vr===null||!vr.has(c))){n=bs(o,n),n=rp(i,n,1),i=_r(i,n,1),n=Rn(),i!==null&&(mt(i,1,n),On(i,n));break}}i=i.return}}function d0(n,i,o){var c=n.pingCache;c!==null&&c.delete(i),i=Rn(),n.pingedLanes|=n.suspendedLanes&o,cn===n&&(_n&o)===o&&(on===4||on===3&&(_n&130023424)===_n&&500>qt()-yc?Xr(n,0):Sc|=o),On(n,i)}function zp(n,i){i===0&&((n.mode&1)===0?i=1:(i=qe,qe<<=1,(qe&130023424)===0&&(qe=4194304)));var o=Rn();n=Vi(n,i),n!==null&&(mt(n,i,o),On(n,o))}function h0(n){var i=n.memoizedState,o=0;i!==null&&(o=i.retryLane),zp(n,o)}function p0(n,i){var o=0;switch(n.tag){case 13:var c=n.stateNode,h=n.memoizedState;h!==null&&(o=h.retryLane);break;case 19:c=n.stateNode;break;default:throw Error(t(314))}c!==null&&c.delete(i),zp(n,o)}var Hp;Hp=function(n,i,o){if(n!==null)if(n.memoizedProps!==i.pendingProps||In.current)Un=!0;else{if((n.lanes&o)===0&&(i.flags&128)===0)return Un=!1,t0(n,i,o);Un=(n.flags&131072)!==0}else Un=!1,Xt&&(i.flags&1048576)!==0&&Sh(i,La,i.index);switch(i.lanes=0,i.tag){case 2:var c=i.type;Ya(n,i),n=i.pendingProps;var h=ys(i,vn.current);Rs(i,o),h=Qu(null,i,c,n,h,o);var x=Ju();return i.flags|=1,typeof h=="object"&&h!==null&&typeof h.render=="function"&&h.$$typeof===void 0?(i.tag=1,i.memoizedState=null,i.updateQueue=null,Nn(c)?(x=!0,Ca(i)):x=!1,i.memoizedState=h.state!==null&&h.state!==void 0?h.state:null,Xu(i),h.updater=Wa,i.stateNode=h,h._reactInternals=i,sc(i,c,n,o),i=uc(null,i,c,!0,x,o)):(i.tag=0,Xt&&x&&Uu(i),An(null,i,h,o),i=i.child),i;case 16:c=i.elementType;e:{switch(Ya(n,i),n=i.pendingProps,h=c._init,c=h(c._payload),i.type=c,h=i.tag=_0(c),n=ui(c,n),h){case 0:i=lc(null,i,c,n,o);break e;case 1:i=hp(null,i,c,n,o);break e;case 11:i=lp(null,i,c,n,o);break e;case 14:i=up(null,i,c,ui(c.type,n),o);break e}throw Error(t(306,c,""))}return i;case 0:return c=i.type,h=i.pendingProps,h=i.elementType===c?h:ui(c,h),lc(n,i,c,h,o);case 1:return c=i.type,h=i.pendingProps,h=i.elementType===c?h:ui(c,h),hp(n,i,c,h,o);case 3:e:{if(pp(i),n===null)throw Error(t(387));c=i.pendingProps,x=i.memoizedState,h=x.element,bh(n,i),Oa(i,c,null,o);var w=i.memoizedState;if(c=w.element,x.isDehydrated)if(x={element:c,isDehydrated:!1,cache:w.cache,pendingSuspenseBoundaries:w.pendingSuspenseBoundaries,transitions:w.transitions},i.updateQueue.baseState=x,i.memoizedState=x,i.flags&256){h=bs(Error(t(423)),i),i=mp(n,i,c,o,h);break e}else if(c!==h){h=bs(Error(t(424)),i),i=mp(n,i,c,o,h);break e}else for(Gn=fr(i.stateNode.containerInfo.firstChild),Vn=i,Xt=!0,li=null,o=Rh(i,null,c,o),i.child=o;o;)o.flags=o.flags&-3|4096,o=o.sibling;else{if(Ts(),c===h){i=Wi(n,i,o);break e}An(n,i,c,o)}i=i.child}return i;case 5:return Dh(i),n===null&&Bu(i),c=i.type,h=i.pendingProps,x=n!==null?n.memoizedProps:null,w=h.children,bu(c,h)?w=null:x!==null&&bu(c,x)&&(i.flags|=32),dp(n,i),An(n,i,w,o),i.child;case 6:return n===null&&Bu(i),null;case 13:return _p(n,i,o);case 4:return Yu(i,i.stateNode.containerInfo),c=i.pendingProps,n===null?i.child=ws(i,null,c,o):An(n,i,c,o),i.child;case 11:return c=i.type,h=i.pendingProps,h=i.elementType===c?h:ui(c,h),lp(n,i,c,h,o);case 7:return An(n,i,i.pendingProps,o),i.child;case 8:return An(n,i,i.pendingProps.children,o),i.child;case 12:return An(n,i,i.pendingProps.children,o),i.child;case 10:e:{if(c=i.type._context,h=i.pendingProps,x=i.memoizedProps,w=h.value,kt(Na,c._currentValue),c._currentValue=w,x!==null)if(ai(x.value,w)){if(x.children===h.children&&!In.current){i=Wi(n,i,o);break e}}else for(x=i.child,x!==null&&(x.return=i);x!==null;){var N=x.dependencies;if(N!==null){w=x.child;for(var B=N.firstContext;B!==null;){if(B.context===c){if(x.tag===1){B=Gi(-1,o&-o),B.tag=2;var ue=x.updateQueue;if(ue!==null){ue=ue.shared;var xe=ue.pending;xe===null?B.next=B:(B.next=xe.next,xe.next=B),ue.pending=B}}x.lanes|=o,B=x.alternate,B!==null&&(B.lanes|=o),Gu(x.return,o,i),N.lanes|=o;break}B=B.next}}else if(x.tag===10)w=x.type===i.type?null:x.child;else if(x.tag===18){if(w=x.return,w===null)throw Error(t(341));w.lanes|=o,N=w.alternate,N!==null&&(N.lanes|=o),Gu(w,o,i),w=x.sibling}else w=x.child;if(w!==null)w.return=x;else for(w=x;w!==null;){if(w===i){w=null;break}if(x=w.sibling,x!==null){x.return=w.return,w=x;break}w=w.return}x=w}An(n,i,h.children,o),i=i.child}return i;case 9:return h=i.type,c=i.pendingProps.children,Rs(i,o),h=$n(h),c=c(h),i.flags|=1,An(n,i,c,o),i.child;case 14:return c=i.type,h=ui(c,i.pendingProps),h=ui(c.type,h),up(n,i,c,h,o);case 15:return cp(n,i,i.type,i.pendingProps,o);case 17:return c=i.type,h=i.pendingProps,h=i.elementType===c?h:ui(c,h),Ya(n,i),i.tag=1,Nn(c)?(n=!0,Ca(i)):n=!1,Rs(i,o),tp(i,c,h),sc(i,c,h,o),uc(null,i,c,!0,n,o);case 19:return vp(n,i,o);case 22:return fp(n,i,o)}throw Error(t(156,i.tag))};function Vp(n,i){return ca(n,i)}function m0(n,i,o,c){this.tag=n,this.key=o,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=i,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=c,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Jn(n,i,o,c){return new m0(n,i,o,c)}function bc(n){return n=n.prototype,!(!n||!n.isReactComponent)}function _0(n){if(typeof n=="function")return bc(n)?1:0;if(n!=null){if(n=n.$$typeof,n===j)return 11;if(n===X)return 14}return 2}function Mr(n,i){var o=n.alternate;return o===null?(o=Jn(n.tag,i,n.key,n.mode),o.elementType=n.elementType,o.type=n.type,o.stateNode=n.stateNode,o.alternate=n,n.alternate=o):(o.pendingProps=i,o.type=n.type,o.flags=0,o.subtreeFlags=0,o.deletions=null),o.flags=n.flags&14680064,o.childLanes=n.childLanes,o.lanes=n.lanes,o.child=n.child,o.memoizedProps=n.memoizedProps,o.memoizedState=n.memoizedState,o.updateQueue=n.updateQueue,i=n.dependencies,o.dependencies=i===null?null:{lanes:i.lanes,firstContext:i.firstContext},o.sibling=n.sibling,o.index=n.index,o.ref=n.ref,o}function il(n,i,o,c,h,x){var w=2;if(c=n,typeof n=="function")bc(n)&&(w=1);else if(typeof n=="string")w=5;else e:switch(n){case F:return qr(o.children,h,x,i);case R:w=8,h|=8;break;case I:return n=Jn(12,o,i,h|2),n.elementType=I,n.lanes=x,n;case re:return n=Jn(13,o,i,h),n.elementType=re,n.lanes=x,n;case ae:return n=Jn(19,o,i,h),n.elementType=ae,n.lanes=x,n;case q:return rl(o,h,x,i);default:if(typeof n=="object"&&n!==null)switch(n.$$typeof){case W:w=10;break e;case O:w=9;break e;case j:w=11;break e;case X:w=14;break e;case Z:w=16,c=null;break e}throw Error(t(130,n==null?n:typeof n,""))}return i=Jn(w,o,i,h),i.elementType=n,i.type=c,i.lanes=x,i}function qr(n,i,o,c){return n=Jn(7,n,c,i),n.lanes=o,n}function rl(n,i,o,c){return n=Jn(22,n,c,i),n.elementType=q,n.lanes=o,n.stateNode={isHidden:!1},n}function Pc(n,i,o){return n=Jn(6,n,null,i),n.lanes=o,n}function Lc(n,i,o){return i=Jn(4,n.children!==null?n.children:[],n.key,i),i.lanes=o,i.stateNode={containerInfo:n.containerInfo,pendingChildren:null,implementation:n.implementation},i}function g0(n,i,o,c,h){this.tag=i,this.containerInfo=n,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=pn(0),this.expirationTimes=pn(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=pn(0),this.identifierPrefix=c,this.onRecoverableError=h,this.mutableSourceEagerHydrationData=null}function Dc(n,i,o,c,h,x,w,N,B){return n=new g0(n,i,o,N,B),i===1?(i=1,x===!0&&(i|=8)):i=0,x=Jn(3,null,null,i),n.current=x,x.stateNode=n,x.memoizedState={element:c,isDehydrated:o,cache:null,transitions:null,pendingSuspenseBoundaries:null},Xu(x),n}function v0(n,i,o){var c=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:D,key:c==null?null:""+c,children:n,containerInfo:i,implementation:o}}function Gp(n){if(!n)return hr;n=n._reactInternals;e:{if(Si(n)!==n||n.tag!==1)throw Error(t(170));var i=n;do{switch(i.tag){case 3:i=i.stateNode.context;break e;case 1:if(Nn(i.type)){i=i.stateNode.__reactInternalMemoizedMergedChildContext;break e}}i=i.return}while(i!==null);throw Error(t(171))}if(n.tag===1){var o=n.type;if(Nn(o))return gh(n,o,i)}return i}function Wp(n,i,o,c,h,x,w,N,B){return n=Dc(o,c,!0,n,h,x,w,N,B),n.context=Gp(null),o=n.current,c=Rn(),h=Sr(o),x=Gi(c,h),x.callback=i??null,_r(o,x,h),n.current.lanes=h,mt(n,h,c),On(n,c),n}function sl(n,i,o,c){var h=i.current,x=Rn(),w=Sr(h);return o=Gp(o),i.context===null?i.context=o:i.pendingContext=o,i=Gi(x,w),i.payload={element:n},c=c===void 0?null:c,c!==null&&(i.callback=c),n=_r(h,i,w),n!==null&&(di(n,h,w,x),Fa(n,h,w)),w}function ol(n){if(n=n.current,!n.child)return null;switch(n.child.tag){case 5:return n.child.stateNode;default:return n.child.stateNode}}function Xp(n,i){if(n=n.memoizedState,n!==null&&n.dehydrated!==null){var o=n.retryLane;n.retryLane=o!==0&&o<i?o:i}}function Ic(n,i){Xp(n,i),(n=n.alternate)&&Xp(n,i)}function x0(){return null}var Yp=typeof reportError=="function"?reportError:function(n){console.error(n)};function Nc(n){this._internalRoot=n}al.prototype.render=Nc.prototype.render=function(n){var i=this._internalRoot;if(i===null)throw Error(t(409));sl(n,i,null,null)},al.prototype.unmount=Nc.prototype.unmount=function(){var n=this._internalRoot;if(n!==null){this._internalRoot=null;var i=n.containerInfo;Wr(function(){sl(null,n,null,null)}),i[Bi]=null}};function al(n){this._internalRoot=n}al.prototype.unstable_scheduleHydration=function(n){if(n){var i=Pt();n={blockedOn:null,target:n,priority:i};for(var o=0;o<lr.length&&i!==0&&i<lr[o].priority;o++);lr.splice(o,0,n),o===0&&Dd(n)}};function Uc(n){return!(!n||n.nodeType!==1&&n.nodeType!==9&&n.nodeType!==11)}function ll(n){return!(!n||n.nodeType!==1&&n.nodeType!==9&&n.nodeType!==11&&(n.nodeType!==8||n.nodeValue!==" react-mount-point-unstable "))}function qp(){}function S0(n,i,o,c,h){if(h){if(typeof c=="function"){var x=c;c=function(){var ue=ol(w);x.call(ue)}}var w=Wp(i,c,n,0,null,!1,!1,"",qp);return n._reactRootContainer=w,n[Bi]=w.current,To(n.nodeType===8?n.parentNode:n),Wr(),w}for(;h=n.lastChild;)n.removeChild(h);if(typeof c=="function"){var N=c;c=function(){var ue=ol(B);N.call(ue)}}var B=Dc(n,0,!1,null,null,!1,!1,"",qp);return n._reactRootContainer=B,n[Bi]=B.current,To(n.nodeType===8?n.parentNode:n),Wr(function(){sl(i,B,o,c)}),B}function ul(n,i,o,c,h){var x=o._reactRootContainer;if(x){var w=x;if(typeof h=="function"){var N=h;h=function(){var B=ol(w);N.call(B)}}sl(i,w,n,h)}else w=S0(o,i,n,h,c);return ol(w)}Rt=function(n){switch(n.tag){case 3:var i=n.stateNode;if(i.current.memoizedState.isDehydrated){var o=St(i.pendingLanes);o!==0&&(Dn(i,o|1),On(i,qt()),(yt&6)===0&&(Ds=qt()+500,pr()))}break;case 13:Wr(function(){var c=Vi(n,1);if(c!==null){var h=Rn();di(c,n,1,h)}}),Ic(n,1)}},Ht=function(n){if(n.tag===13){var i=Vi(n,134217728);if(i!==null){var o=Rn();di(i,n,134217728,o)}Ic(n,134217728)}},si=function(n){if(n.tag===13){var i=Sr(n),o=Vi(n,i);if(o!==null){var c=Rn();di(o,n,i,c)}Ic(n,i)}},Pt=function(){return _t},oi=function(n,i){var o=_t;try{return _t=n,i()}finally{_t=o}},rt=function(n,i,o){switch(i){case"input":if(Yt(n,o),i=o.name,o.type==="radio"&&i!=null){for(o=n;o.parentNode;)o=o.parentNode;for(o=o.querySelectorAll("input[name="+JSON.stringify(""+i)+'][type="radio"]'),i=0;i<o.length;i++){var c=o[i];if(c!==n&&c.form===n.form){var h=Aa(c);if(!h)throw Error(t(90));Ut(c),Yt(c,h)}}}break;case"textarea":Ne(n,o);break;case"select":i=o.value,i!=null&&Ot(n,!!o.multiple,i,!1)}},Ce=Ac,ge=Wr;var y0={usingClientEntryPoint:!1,Events:[Ro,xs,Aa,fe,Oe,Ac]},Ho={findFiberByHostInstance:Fr,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},M0={bundleType:Ho.bundleType,version:Ho.version,rendererPackageName:Ho.rendererPackageName,rendererConfig:Ho.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:L.ReactCurrentDispatcher,findHostInstanceByFiber:function(n){return n=ao(n),n===null?null:n.stateNode},findFiberByHostInstance:Ho.findFiberByHostInstance||x0,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var cl=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!cl.isDisabled&&cl.supportsFiber)try{ee=cl.inject(M0),be=cl}catch{}}return Bn.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=y0,Bn.createPortal=function(n,i){var o=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!Uc(i))throw Error(t(200));return v0(n,i,null,o)},Bn.createRoot=function(n,i){if(!Uc(n))throw Error(t(299));var o=!1,c="",h=Yp;return i!=null&&(i.unstable_strictMode===!0&&(o=!0),i.identifierPrefix!==void 0&&(c=i.identifierPrefix),i.onRecoverableError!==void 0&&(h=i.onRecoverableError)),i=Dc(n,1,!1,null,null,o,!1,c,h),n[Bi]=i.current,To(n.nodeType===8?n.parentNode:n),new Nc(i)},Bn.findDOMNode=function(n){if(n==null)return null;if(n.nodeType===1)return n;var i=n._reactInternals;if(i===void 0)throw typeof n.render=="function"?Error(t(188)):(n=Object.keys(n).join(","),Error(t(268,n)));return n=ao(i),n=n===null?null:n.stateNode,n},Bn.flushSync=function(n){return Wr(n)},Bn.hydrate=function(n,i,o){if(!ll(i))throw Error(t(200));return ul(null,n,i,!0,o)},Bn.hydrateRoot=function(n,i,o){if(!Uc(n))throw Error(t(405));var c=o!=null&&o.hydratedSources||null,h=!1,x="",w=Yp;if(o!=null&&(o.unstable_strictMode===!0&&(h=!0),o.identifierPrefix!==void 0&&(x=o.identifierPrefix),o.onRecoverableError!==void 0&&(w=o.onRecoverableError)),i=Wp(i,null,n,1,o??null,h,!1,x,w),n[Bi]=i.current,To(n),c)for(n=0;n<c.length;n++)o=c[n],h=o._getVersion,h=h(o._source),i.mutableSourceEagerHydrationData==null?i.mutableSourceEagerHydrationData=[o,h]:i.mutableSourceEagerHydrationData.push(o,h);return new al(i)},Bn.render=function(n,i,o){if(!ll(i))throw Error(t(200));return ul(null,n,i,!1,o)},Bn.unmountComponentAtNode=function(n){if(!ll(n))throw Error(t(40));return n._reactRootContainer?(Wr(function(){ul(null,null,n,!1,function(){n._reactRootContainer=null,n[Bi]=null})}),!0):!1},Bn.unstable_batchedUpdates=Ac,Bn.unstable_renderSubtreeIntoContainer=function(n,i,o,c){if(!ll(o))throw Error(t(200));if(n==null||n._reactInternals===void 0)throw Error(t(38));return ul(n,i,o,!1,c)},Bn.version="18.3.1-next-f1338f8080-20240426",Bn}var tm;function P0(){if(tm)return Bc.exports;tm=1;function s(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(s)}catch(e){console.error(e)}}return s(),Bc.exports=b0(),Bc.exports}var nm;function L0(){if(nm)return fl;nm=1;var s=P0();return fl.createRoot=s.createRoot,fl.hydrateRoot=s.hydrateRoot,fl}var D0=L0();/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const pd="184",I0=0,im=1,N0=2,kl=1,U0=2,Zo=3,Dr=0,kn=1,Zi=2,Ji=0,Ks=1,Xl=2,rm=3,sm=4,F0=5,es=100,O0=101,B0=102,k0=103,z0=104,H0=200,V0=201,G0=202,W0=203,Mf=204,Ef=205,X0=206,Y0=207,q0=208,j0=209,K0=210,$0=211,Z0=212,Q0=213,J0=214,Tf=0,wf=1,Af=2,Zs=3,Rf=4,Cf=5,bf=6,Pf=7,a_=0,ev=1,tv=2,Di=0,l_=1,u_=2,c_=3,f_=4,d_=5,h_=6,p_=7,m_=300,ss=301,Qs=302,Hc=303,Vc=304,Jl=306,Lf=1e3,Qi=1001,Df=1002,gn=1003,nv=1004,dl=1005,Tn=1006,Gc=1007,ns=1008,ii=1009,__=1010,g_=1011,na=1012,md=1013,Ni=1014,Pi=1015,nr=1016,_d=1017,gd=1018,ia=1020,v_=35902,x_=35899,S_=1021,y_=1022,gi=1023,ir=1026,is=1027,M_=1028,vd=1029,os=1030,xd=1031,Sd=1033,zl=33776,Hl=33777,Vl=33778,Gl=33779,If=35840,Nf=35841,Uf=35842,Ff=35843,Of=36196,Bf=37492,kf=37496,zf=37488,Hf=37489,Yl=37490,Vf=37491,Gf=37808,Wf=37809,Xf=37810,Yf=37811,qf=37812,jf=37813,Kf=37814,$f=37815,Zf=37816,Qf=37817,Jf=37818,ed=37819,td=37820,nd=37821,id=36492,rd=36494,sd=36495,od=36283,ad=36284,ql=36285,ld=36286,iv=3200,om=0,rv=1,Pr="",ti="srgb",jl="srgb-linear",Kl="linear",Lt="srgb",Ns=7680,am=519,sv=512,ov=513,av=514,yd=515,lv=516,uv=517,Md=518,cv=519,lm=35044,um="300 es",Li=2e3,$l=2001;function fv(s){for(let e=s.length-1;e>=0;--e)if(s[e]>=65535)return!0;return!1}function Zl(s){return document.createElementNS("http://www.w3.org/1999/xhtml",s)}function dv(){const s=Zl("canvas");return s.style.display="block",s}const cm={};function fm(...s){const e="THREE."+s.shift();console.log(e,...s)}function E_(s){const e=s[0];if(typeof e=="string"&&e.startsWith("TSL:")){const t=s[1];t&&t.isStackTrace?s[0]+=" "+t.getLocation():s[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return s}function tt(...s){s=E_(s);const e="THREE."+s.shift();{const t=s[0];t&&t.isStackTrace?console.warn(t.getError(e)):console.warn(e,...s)}}function Mt(...s){s=E_(s);const e="THREE."+s.shift();{const t=s[0];t&&t.isStackTrace?console.error(t.getError(e)):console.error(e,...s)}}function ud(...s){const e=s.join(" ");e in cm||(cm[e]=!0,tt(...s))}function hv(s,e,t){return new Promise(function(r,a){function l(){switch(s.clientWaitSync(e,s.SYNC_FLUSH_COMMANDS_BIT,0)){case s.WAIT_FAILED:a();break;case s.TIMEOUT_EXPIRED:setTimeout(l,t);break;default:r()}}setTimeout(l,t)})}const pv={[Tf]:wf,[Af]:bf,[Rf]:Pf,[Zs]:Cf,[wf]:Tf,[bf]:Af,[Pf]:Rf,[Cf]:Zs};class ls{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const r=this._listeners;r[e]===void 0&&(r[e]=[]),r[e].indexOf(t)===-1&&r[e].push(t)}hasEventListener(e,t){const r=this._listeners;return r===void 0?!1:r[e]!==void 0&&r[e].indexOf(t)!==-1}removeEventListener(e,t){const r=this._listeners;if(r===void 0)return;const a=r[e];if(a!==void 0){const l=a.indexOf(t);l!==-1&&a.splice(l,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const r=t[e.type];if(r!==void 0){e.target=this;const a=r.slice(0);for(let l=0,d=a.length;l<d;l++)a[l].call(this,e);e.target=null}}}const Mn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let dm=1234567;const ea=Math.PI/180,ra=180/Math.PI;function no(){const s=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,r=Math.random()*4294967295|0;return(Mn[s&255]+Mn[s>>8&255]+Mn[s>>16&255]+Mn[s>>24&255]+"-"+Mn[e&255]+Mn[e>>8&255]+"-"+Mn[e>>16&15|64]+Mn[e>>24&255]+"-"+Mn[t&63|128]+Mn[t>>8&255]+"-"+Mn[t>>16&255]+Mn[t>>24&255]+Mn[r&255]+Mn[r>>8&255]+Mn[r>>16&255]+Mn[r>>24&255]).toLowerCase()}function vt(s,e,t){return Math.max(e,Math.min(t,s))}function Ed(s,e){return(s%e+e)%e}function mv(s,e,t,r,a){return r+(s-e)*(a-r)/(t-e)}function _v(s,e,t){return s!==e?(t-s)/(e-s):0}function ta(s,e,t){return(1-t)*s+t*e}function gv(s,e,t,r){return ta(s,e,1-Math.exp(-t*r))}function vv(s,e=1){return e-Math.abs(Ed(s,e*2)-e)}function xv(s,e,t){return s<=e?0:s>=t?1:(s=(s-e)/(t-e),s*s*(3-2*s))}function Sv(s,e,t){return s<=e?0:s>=t?1:(s=(s-e)/(t-e),s*s*s*(s*(s*6-15)+10))}function yv(s,e){return s+Math.floor(Math.random()*(e-s+1))}function Mv(s,e){return s+Math.random()*(e-s)}function Ev(s){return s*(.5-Math.random())}function Tv(s){s!==void 0&&(dm=s);let e=dm+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function wv(s){return s*ea}function Av(s){return s*ra}function Rv(s){return(s&s-1)===0&&s!==0}function Cv(s){return Math.pow(2,Math.ceil(Math.log(s)/Math.LN2))}function bv(s){return Math.pow(2,Math.floor(Math.log(s)/Math.LN2))}function Pv(s,e,t,r,a){const l=Math.cos,d=Math.sin,m=l(t/2),g=d(t/2),_=l((e+r)/2),M=d((e+r)/2),u=l((e-r)/2),f=d((e-r)/2),p=l((r-e)/2),y=d((r-e)/2);switch(a){case"XYX":s.set(m*M,g*u,g*f,m*_);break;case"YZY":s.set(g*f,m*M,g*u,m*_);break;case"ZXZ":s.set(g*u,g*f,m*M,m*_);break;case"XZX":s.set(m*M,g*y,g*p,m*_);break;case"YXY":s.set(g*p,m*M,g*y,m*_);break;case"ZYZ":s.set(g*y,g*p,m*M,m*_);break;default:tt("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+a)}}function js(s,e){switch(e.constructor){case Float32Array:return s;case Uint32Array:return s/4294967295;case Uint16Array:return s/65535;case Uint8Array:return s/255;case Int32Array:return Math.max(s/2147483647,-1);case Int16Array:return Math.max(s/32767,-1);case Int8Array:return Math.max(s/127,-1);default:throw new Error("Invalid component type.")}}function Cn(s,e){switch(e.constructor){case Float32Array:return s;case Uint32Array:return Math.round(s*4294967295);case Uint16Array:return Math.round(s*65535);case Uint8Array:return Math.round(s*255);case Int32Array:return Math.round(s*2147483647);case Int16Array:return Math.round(s*32767);case Int8Array:return Math.round(s*127);default:throw new Error("Invalid component type.")}}const Lv={DEG2RAD:ea,RAD2DEG:ra,generateUUID:no,clamp:vt,euclideanModulo:Ed,mapLinear:mv,inverseLerp:_v,lerp:ta,damp:gv,pingpong:vv,smoothstep:xv,smootherstep:Sv,randInt:yv,randFloat:Mv,randFloatSpread:Ev,seededRandom:Tv,degToRad:wv,radToDeg:Av,isPowerOfTwo:Rv,ceilPowerOfTwo:Cv,floorPowerOfTwo:bv,setQuaternionFromProperEuler:Pv,normalize:Cn,denormalize:js},Ad=class Ad{constructor(e=0,t=0){this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,r=this.y,a=e.elements;return this.x=a[0]*t+a[3]*r+a[6],this.y=a[1]*t+a[4]*r+a[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=vt(this.x,e.x,t.x),this.y=vt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=vt(this.x,e,t),this.y=vt(this.y,e,t),this}clampLength(e,t){const r=this.length();return this.divideScalar(r||1).multiplyScalar(vt(r,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const r=this.dot(e)/t;return Math.acos(vt(r,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,r=this.y-e.y;return t*t+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,r){return this.x=e.x+(t.x-e.x)*r,this.y=e.y+(t.y-e.y)*r,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const r=Math.cos(t),a=Math.sin(t),l=this.x-e.x,d=this.y-e.y;return this.x=l*r-d*a+e.x,this.y=l*a+d*r+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};Ad.prototype.isVector2=!0;let It=Ad;class io{constructor(e=0,t=0,r=0,a=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=r,this._w=a}static slerpFlat(e,t,r,a,l,d,m){let g=r[a+0],_=r[a+1],M=r[a+2],u=r[a+3],f=l[d+0],p=l[d+1],y=l[d+2],E=l[d+3];if(u!==E||g!==f||_!==p||M!==y){let S=g*f+_*p+M*y+u*E;S<0&&(f=-f,p=-p,y=-y,E=-E,S=-S);let v=1-m;if(S<.9995){const A=Math.acos(S),P=Math.sin(A);v=Math.sin(v*A)/P,m=Math.sin(m*A)/P,g=g*v+f*m,_=_*v+p*m,M=M*v+y*m,u=u*v+E*m}else{g=g*v+f*m,_=_*v+p*m,M=M*v+y*m,u=u*v+E*m;const A=1/Math.sqrt(g*g+_*_+M*M+u*u);g*=A,_*=A,M*=A,u*=A}}e[t]=g,e[t+1]=_,e[t+2]=M,e[t+3]=u}static multiplyQuaternionsFlat(e,t,r,a,l,d){const m=r[a],g=r[a+1],_=r[a+2],M=r[a+3],u=l[d],f=l[d+1],p=l[d+2],y=l[d+3];return e[t]=m*y+M*u+g*p-_*f,e[t+1]=g*y+M*f+_*u-m*p,e[t+2]=_*y+M*p+m*f-g*u,e[t+3]=M*y-m*u-g*f-_*p,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,r,a){return this._x=e,this._y=t,this._z=r,this._w=a,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const r=e._x,a=e._y,l=e._z,d=e._order,m=Math.cos,g=Math.sin,_=m(r/2),M=m(a/2),u=m(l/2),f=g(r/2),p=g(a/2),y=g(l/2);switch(d){case"XYZ":this._x=f*M*u+_*p*y,this._y=_*p*u-f*M*y,this._z=_*M*y+f*p*u,this._w=_*M*u-f*p*y;break;case"YXZ":this._x=f*M*u+_*p*y,this._y=_*p*u-f*M*y,this._z=_*M*y-f*p*u,this._w=_*M*u+f*p*y;break;case"ZXY":this._x=f*M*u-_*p*y,this._y=_*p*u+f*M*y,this._z=_*M*y+f*p*u,this._w=_*M*u-f*p*y;break;case"ZYX":this._x=f*M*u-_*p*y,this._y=_*p*u+f*M*y,this._z=_*M*y-f*p*u,this._w=_*M*u+f*p*y;break;case"YZX":this._x=f*M*u+_*p*y,this._y=_*p*u+f*M*y,this._z=_*M*y-f*p*u,this._w=_*M*u-f*p*y;break;case"XZY":this._x=f*M*u-_*p*y,this._y=_*p*u-f*M*y,this._z=_*M*y+f*p*u,this._w=_*M*u+f*p*y;break;default:tt("Quaternion: .setFromEuler() encountered an unknown order: "+d)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const r=t/2,a=Math.sin(r);return this._x=e.x*a,this._y=e.y*a,this._z=e.z*a,this._w=Math.cos(r),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,r=t[0],a=t[4],l=t[8],d=t[1],m=t[5],g=t[9],_=t[2],M=t[6],u=t[10],f=r+m+u;if(f>0){const p=.5/Math.sqrt(f+1);this._w=.25/p,this._x=(M-g)*p,this._y=(l-_)*p,this._z=(d-a)*p}else if(r>m&&r>u){const p=2*Math.sqrt(1+r-m-u);this._w=(M-g)/p,this._x=.25*p,this._y=(a+d)/p,this._z=(l+_)/p}else if(m>u){const p=2*Math.sqrt(1+m-r-u);this._w=(l-_)/p,this._x=(a+d)/p,this._y=.25*p,this._z=(g+M)/p}else{const p=2*Math.sqrt(1+u-r-m);this._w=(d-a)/p,this._x=(l+_)/p,this._y=(g+M)/p,this._z=.25*p}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let r=e.dot(t)+1;return r<1e-8?(r=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=r):(this._x=0,this._y=-e.z,this._z=e.y,this._w=r)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=r),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(vt(this.dot(e),-1,1)))}rotateTowards(e,t){const r=this.angleTo(e);if(r===0)return this;const a=Math.min(1,t/r);return this.slerp(e,a),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const r=e._x,a=e._y,l=e._z,d=e._w,m=t._x,g=t._y,_=t._z,M=t._w;return this._x=r*M+d*m+a*_-l*g,this._y=a*M+d*g+l*m-r*_,this._z=l*M+d*_+r*g-a*m,this._w=d*M-r*m-a*g-l*_,this._onChangeCallback(),this}slerp(e,t){let r=e._x,a=e._y,l=e._z,d=e._w,m=this.dot(e);m<0&&(r=-r,a=-a,l=-l,d=-d,m=-m);let g=1-t;if(m<.9995){const _=Math.acos(m),M=Math.sin(_);g=Math.sin(g*_)/M,t=Math.sin(t*_)/M,this._x=this._x*g+r*t,this._y=this._y*g+a*t,this._z=this._z*g+l*t,this._w=this._w*g+d*t,this._onChangeCallback()}else this._x=this._x*g+r*t,this._y=this._y*g+a*t,this._z=this._z*g+l*t,this._w=this._w*g+d*t,this.normalize();return this}slerpQuaternions(e,t,r){return this.copy(e).slerp(t,r)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),r=Math.random(),a=Math.sqrt(1-r),l=Math.sqrt(r);return this.set(a*Math.sin(e),a*Math.cos(e),l*Math.sin(t),l*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}const Rd=class Rd{constructor(e=0,t=0,r=0){this.x=e,this.y=t,this.z=r}set(e,t,r){return r===void 0&&(r=this.z),this.x=e,this.y=t,this.z=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(hm.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(hm.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,r=this.y,a=this.z,l=e.elements;return this.x=l[0]*t+l[3]*r+l[6]*a,this.y=l[1]*t+l[4]*r+l[7]*a,this.z=l[2]*t+l[5]*r+l[8]*a,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,r=this.y,a=this.z,l=e.elements,d=1/(l[3]*t+l[7]*r+l[11]*a+l[15]);return this.x=(l[0]*t+l[4]*r+l[8]*a+l[12])*d,this.y=(l[1]*t+l[5]*r+l[9]*a+l[13])*d,this.z=(l[2]*t+l[6]*r+l[10]*a+l[14])*d,this}applyQuaternion(e){const t=this.x,r=this.y,a=this.z,l=e.x,d=e.y,m=e.z,g=e.w,_=2*(d*a-m*r),M=2*(m*t-l*a),u=2*(l*r-d*t);return this.x=t+g*_+d*u-m*M,this.y=r+g*M+m*_-l*u,this.z=a+g*u+l*M-d*_,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,r=this.y,a=this.z,l=e.elements;return this.x=l[0]*t+l[4]*r+l[8]*a,this.y=l[1]*t+l[5]*r+l[9]*a,this.z=l[2]*t+l[6]*r+l[10]*a,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=vt(this.x,e.x,t.x),this.y=vt(this.y,e.y,t.y),this.z=vt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=vt(this.x,e,t),this.y=vt(this.y,e,t),this.z=vt(this.z,e,t),this}clampLength(e,t){const r=this.length();return this.divideScalar(r||1).multiplyScalar(vt(r,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,r){return this.x=e.x+(t.x-e.x)*r,this.y=e.y+(t.y-e.y)*r,this.z=e.z+(t.z-e.z)*r,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const r=e.x,a=e.y,l=e.z,d=t.x,m=t.y,g=t.z;return this.x=a*g-l*m,this.y=l*d-r*g,this.z=r*m-a*d,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const r=e.dot(this)/t;return this.copy(e).multiplyScalar(r)}projectOnPlane(e){return Wc.copy(this).projectOnVector(e),this.sub(Wc)}reflect(e){return this.sub(Wc.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const r=this.dot(e)/t;return Math.acos(vt(r,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,r=this.y-e.y,a=this.z-e.z;return t*t+r*r+a*a}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,r){const a=Math.sin(t)*e;return this.x=a*Math.sin(r),this.y=Math.cos(t)*e,this.z=a*Math.cos(r),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,r){return this.x=e*Math.sin(t),this.y=r,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),r=this.setFromMatrixColumn(e,1).length(),a=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=r,this.z=a,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,r=Math.sqrt(1-t*t);return this.x=r*Math.cos(e),this.y=t,this.z=r*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};Rd.prototype.isVector3=!0;let oe=Rd;const Wc=new oe,hm=new io,Cd=class Cd{constructor(e,t,r,a,l,d,m,g,_){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,r,a,l,d,m,g,_)}set(e,t,r,a,l,d,m,g,_){const M=this.elements;return M[0]=e,M[1]=a,M[2]=m,M[3]=t,M[4]=l,M[5]=g,M[6]=r,M[7]=d,M[8]=_,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,r=e.elements;return t[0]=r[0],t[1]=r[1],t[2]=r[2],t[3]=r[3],t[4]=r[4],t[5]=r[5],t[6]=r[6],t[7]=r[7],t[8]=r[8],this}extractBasis(e,t,r){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),r.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const r=e.elements,a=t.elements,l=this.elements,d=r[0],m=r[3],g=r[6],_=r[1],M=r[4],u=r[7],f=r[2],p=r[5],y=r[8],E=a[0],S=a[3],v=a[6],A=a[1],P=a[4],L=a[7],z=a[2],D=a[5],F=a[8];return l[0]=d*E+m*A+g*z,l[3]=d*S+m*P+g*D,l[6]=d*v+m*L+g*F,l[1]=_*E+M*A+u*z,l[4]=_*S+M*P+u*D,l[7]=_*v+M*L+u*F,l[2]=f*E+p*A+y*z,l[5]=f*S+p*P+y*D,l[8]=f*v+p*L+y*F,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],r=e[1],a=e[2],l=e[3],d=e[4],m=e[5],g=e[6],_=e[7],M=e[8];return t*d*M-t*m*_-r*l*M+r*m*g+a*l*_-a*d*g}invert(){const e=this.elements,t=e[0],r=e[1],a=e[2],l=e[3],d=e[4],m=e[5],g=e[6],_=e[7],M=e[8],u=M*d-m*_,f=m*g-M*l,p=_*l-d*g,y=t*u+r*f+a*p;if(y===0)return this.set(0,0,0,0,0,0,0,0,0);const E=1/y;return e[0]=u*E,e[1]=(a*_-M*r)*E,e[2]=(m*r-a*d)*E,e[3]=f*E,e[4]=(M*t-a*g)*E,e[5]=(a*l-m*t)*E,e[6]=p*E,e[7]=(r*g-_*t)*E,e[8]=(d*t-r*l)*E,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,r,a,l,d,m){const g=Math.cos(l),_=Math.sin(l);return this.set(r*g,r*_,-r*(g*d+_*m)+d+e,-a*_,a*g,-a*(-_*d+g*m)+m+t,0,0,1),this}scale(e,t){return this.premultiply(Xc.makeScale(e,t)),this}rotate(e){return this.premultiply(Xc.makeRotation(-e)),this}translate(e,t){return this.premultiply(Xc.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),r=Math.sin(e);return this.set(t,-r,0,r,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,r=e.elements;for(let a=0;a<9;a++)if(t[a]!==r[a])return!1;return!0}fromArray(e,t=0){for(let r=0;r<9;r++)this.elements[r]=e[r+t];return this}toArray(e=[],t=0){const r=this.elements;return e[t]=r[0],e[t+1]=r[1],e[t+2]=r[2],e[t+3]=r[3],e[t+4]=r[4],e[t+5]=r[5],e[t+6]=r[6],e[t+7]=r[7],e[t+8]=r[8],e}clone(){return new this.constructor().fromArray(this.elements)}};Cd.prototype.isMatrix3=!0;let lt=Cd;const Xc=new lt,pm=new lt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),mm=new lt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Dv(){const s={enabled:!0,workingColorSpace:jl,spaces:{},convert:function(a,l,d){return this.enabled===!1||l===d||!l||!d||(this.spaces[l].transfer===Lt&&(a.r=er(a.r),a.g=er(a.g),a.b=er(a.b)),this.spaces[l].primaries!==this.spaces[d].primaries&&(a.applyMatrix3(this.spaces[l].toXYZ),a.applyMatrix3(this.spaces[d].fromXYZ)),this.spaces[d].transfer===Lt&&(a.r=$s(a.r),a.g=$s(a.g),a.b=$s(a.b))),a},workingToColorSpace:function(a,l){return this.convert(a,this.workingColorSpace,l)},colorSpaceToWorking:function(a,l){return this.convert(a,l,this.workingColorSpace)},getPrimaries:function(a){return this.spaces[a].primaries},getTransfer:function(a){return a===Pr?Kl:this.spaces[a].transfer},getToneMappingMode:function(a){return this.spaces[a].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(a,l=this.workingColorSpace){return a.fromArray(this.spaces[l].luminanceCoefficients)},define:function(a){Object.assign(this.spaces,a)},_getMatrix:function(a,l,d){return a.copy(this.spaces[l].toXYZ).multiply(this.spaces[d].fromXYZ)},_getDrawingBufferColorSpace:function(a){return this.spaces[a].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(a=this.workingColorSpace){return this.spaces[a].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(a,l){return ud("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),s.workingToColorSpace(a,l)},toWorkingColorSpace:function(a,l){return ud("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),s.colorSpaceToWorking(a,l)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],r=[.3127,.329];return s.define({[jl]:{primaries:e,whitePoint:r,transfer:Kl,toXYZ:pm,fromXYZ:mm,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:ti},outputColorSpaceConfig:{drawingBufferColorSpace:ti}},[ti]:{primaries:e,whitePoint:r,transfer:Lt,toXYZ:pm,fromXYZ:mm,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:ti}}}),s}const xt=Dv();function er(s){return s<.04045?s*.0773993808:Math.pow(s*.9478672986+.0521327014,2.4)}function $s(s){return s<.0031308?s*12.92:1.055*Math.pow(s,.41666)-.055}let Us;class Iv{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let r;if(e instanceof HTMLCanvasElement)r=e;else{Us===void 0&&(Us=Zl("canvas")),Us.width=e.width,Us.height=e.height;const a=Us.getContext("2d");e instanceof ImageData?a.putImageData(e,0,0):a.drawImage(e,0,0,e.width,e.height),r=Us}return r.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Zl("canvas");t.width=e.width,t.height=e.height;const r=t.getContext("2d");r.drawImage(e,0,0,e.width,e.height);const a=r.getImageData(0,0,e.width,e.height),l=a.data;for(let d=0;d<l.length;d++)l[d]=er(l[d]/255)*255;return r.putImageData(a,0,0),t}else if(e.data){const t=e.data.slice(0);for(let r=0;r<t.length;r++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[r]=Math.floor(er(t[r]/255)*255):t[r]=er(t[r]);return{data:t,width:e.width,height:e.height}}else return tt("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let Nv=0;class Td{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:Nv++}),this.uuid=no(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayWidth,t.displayHeight,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const r={uuid:this.uuid,url:""},a=this.data;if(a!==null){let l;if(Array.isArray(a)){l=[];for(let d=0,m=a.length;d<m;d++)a[d].isDataTexture?l.push(Yc(a[d].image)):l.push(Yc(a[d]))}else l=Yc(a);r.url=l}return t||(e.images[this.uuid]=r),r}}function Yc(s){return typeof HTMLImageElement<"u"&&s instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&s instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&s instanceof ImageBitmap?Iv.getDataURL(s):s.data?{data:Array.from(s.data),width:s.width,height:s.height,type:s.data.constructor.name}:(tt("Texture: Unable to serialize Texture."),{})}let Uv=0;const qc=new oe;class Pn extends ls{constructor(e=Pn.DEFAULT_IMAGE,t=Pn.DEFAULT_MAPPING,r=Qi,a=Qi,l=Tn,d=ns,m=gi,g=ii,_=Pn.DEFAULT_ANISOTROPY,M=Pr){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Uv++}),this.uuid=no(),this.name="",this.source=new Td(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=r,this.wrapT=a,this.magFilter=l,this.minFilter=d,this.anisotropy=_,this.format=m,this.internalFormat=null,this.type=g,this.offset=new It(0,0),this.repeat=new It(1,1),this.center=new It(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new lt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=M,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(qc).x}get height(){return this.source.getSize(qc).y}get depth(){return this.source.getSize(qc).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const r=e[t];if(r===void 0){tt(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const a=this[t];if(a===void 0){tt(`Texture.setValues(): property '${t}' does not exist.`);continue}a&&r&&a.isVector2&&r.isVector2||a&&r&&a.isVector3&&r.isVector3||a&&r&&a.isMatrix3&&r.isMatrix3?a.copy(r):this[t]=r}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const r={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(r.userData=this.userData),t||(e.textures[this.uuid]=r),r}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==m_)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Lf:e.x=e.x-Math.floor(e.x);break;case Qi:e.x=e.x<0?0:1;break;case Df:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Lf:e.y=e.y-Math.floor(e.y);break;case Qi:e.y=e.y<0?0:1;break;case Df:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Pn.DEFAULT_IMAGE=null;Pn.DEFAULT_MAPPING=m_;Pn.DEFAULT_ANISOTROPY=1;const bd=class bd{constructor(e=0,t=0,r=0,a=1){this.x=e,this.y=t,this.z=r,this.w=a}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,r,a){return this.x=e,this.y=t,this.z=r,this.w=a,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,r=this.y,a=this.z,l=this.w,d=e.elements;return this.x=d[0]*t+d[4]*r+d[8]*a+d[12]*l,this.y=d[1]*t+d[5]*r+d[9]*a+d[13]*l,this.z=d[2]*t+d[6]*r+d[10]*a+d[14]*l,this.w=d[3]*t+d[7]*r+d[11]*a+d[15]*l,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,r,a,l;const g=e.elements,_=g[0],M=g[4],u=g[8],f=g[1],p=g[5],y=g[9],E=g[2],S=g[6],v=g[10];if(Math.abs(M-f)<.01&&Math.abs(u-E)<.01&&Math.abs(y-S)<.01){if(Math.abs(M+f)<.1&&Math.abs(u+E)<.1&&Math.abs(y+S)<.1&&Math.abs(_+p+v-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const P=(_+1)/2,L=(p+1)/2,z=(v+1)/2,D=(M+f)/4,F=(u+E)/4,R=(y+S)/4;return P>L&&P>z?P<.01?(r=0,a=.707106781,l=.707106781):(r=Math.sqrt(P),a=D/r,l=F/r):L>z?L<.01?(r=.707106781,a=0,l=.707106781):(a=Math.sqrt(L),r=D/a,l=R/a):z<.01?(r=.707106781,a=.707106781,l=0):(l=Math.sqrt(z),r=F/l,a=R/l),this.set(r,a,l,t),this}let A=Math.sqrt((S-y)*(S-y)+(u-E)*(u-E)+(f-M)*(f-M));return Math.abs(A)<.001&&(A=1),this.x=(S-y)/A,this.y=(u-E)/A,this.z=(f-M)/A,this.w=Math.acos((_+p+v-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=vt(this.x,e.x,t.x),this.y=vt(this.y,e.y,t.y),this.z=vt(this.z,e.z,t.z),this.w=vt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=vt(this.x,e,t),this.y=vt(this.y,e,t),this.z=vt(this.z,e,t),this.w=vt(this.w,e,t),this}clampLength(e,t){const r=this.length();return this.divideScalar(r||1).multiplyScalar(vt(r,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,r){return this.x=e.x+(t.x-e.x)*r,this.y=e.y+(t.y-e.y)*r,this.z=e.z+(t.z-e.z)*r,this.w=e.w+(t.w-e.w)*r,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};bd.prototype.isVector4=!0;let Jt=bd;class Fv extends ls{constructor(e=1,t=1,r={}){super(),r=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Tn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},r),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=r.depth,this.scissor=new Jt(0,0,e,t),this.scissorTest=!1,this.viewport=new Jt(0,0,e,t),this.textures=[];const a={width:e,height:t,depth:r.depth},l=new Pn(a),d=r.count;for(let m=0;m<d;m++)this.textures[m]=l.clone(),this.textures[m].isRenderTargetTexture=!0,this.textures[m].renderTarget=this;this._setTextureOptions(r),this.depthBuffer=r.depthBuffer,this.stencilBuffer=r.stencilBuffer,this.resolveDepthBuffer=r.resolveDepthBuffer,this.resolveStencilBuffer=r.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=r.depthTexture,this.samples=r.samples,this.multiview=r.multiview}_setTextureOptions(e={}){const t={minFilter:Tn,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let r=0;r<this.textures.length;r++)this.textures[r].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,r=1){if(this.width!==e||this.height!==t||this.depth!==r){this.width=e,this.height=t,this.depth=r;for(let a=0,l=this.textures.length;a<l;a++)this.textures[a].image.width=e,this.textures[a].image.height=t,this.textures[a].image.depth=r,this.textures[a].isData3DTexture!==!0&&(this.textures[a].isArrayTexture=this.textures[a].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,r=e.textures.length;t<r;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const a=Object.assign({},e.textures[t].image);this.textures[t].source=new Td(a)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Ii extends Fv{constructor(e=1,t=1,r={}){super(e,t,r),this.isWebGLRenderTarget=!0}}class T_ extends Pn{constructor(e=null,t=1,r=1,a=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:r,depth:a},this.magFilter=gn,this.minFilter=gn,this.wrapR=Qi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class Ov extends Pn{constructor(e=null,t=1,r=1,a=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:r,depth:a},this.magFilter=gn,this.minFilter=gn,this.wrapR=Qi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const Ql=class Ql{constructor(e,t,r,a,l,d,m,g,_,M,u,f,p,y,E,S){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,r,a,l,d,m,g,_,M,u,f,p,y,E,S)}set(e,t,r,a,l,d,m,g,_,M,u,f,p,y,E,S){const v=this.elements;return v[0]=e,v[4]=t,v[8]=r,v[12]=a,v[1]=l,v[5]=d,v[9]=m,v[13]=g,v[2]=_,v[6]=M,v[10]=u,v[14]=f,v[3]=p,v[7]=y,v[11]=E,v[15]=S,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Ql().fromArray(this.elements)}copy(e){const t=this.elements,r=e.elements;return t[0]=r[0],t[1]=r[1],t[2]=r[2],t[3]=r[3],t[4]=r[4],t[5]=r[5],t[6]=r[6],t[7]=r[7],t[8]=r[8],t[9]=r[9],t[10]=r[10],t[11]=r[11],t[12]=r[12],t[13]=r[13],t[14]=r[14],t[15]=r[15],this}copyPosition(e){const t=this.elements,r=e.elements;return t[12]=r[12],t[13]=r[13],t[14]=r[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,r){return this.determinant()===0?(e.set(1,0,0),t.set(0,1,0),r.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),r.setFromMatrixColumn(this,2),this)}makeBasis(e,t,r){return this.set(e.x,t.x,r.x,0,e.y,t.y,r.y,0,e.z,t.z,r.z,0,0,0,0,1),this}extractRotation(e){if(e.determinant()===0)return this.identity();const t=this.elements,r=e.elements,a=1/Fs.setFromMatrixColumn(e,0).length(),l=1/Fs.setFromMatrixColumn(e,1).length(),d=1/Fs.setFromMatrixColumn(e,2).length();return t[0]=r[0]*a,t[1]=r[1]*a,t[2]=r[2]*a,t[3]=0,t[4]=r[4]*l,t[5]=r[5]*l,t[6]=r[6]*l,t[7]=0,t[8]=r[8]*d,t[9]=r[9]*d,t[10]=r[10]*d,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,r=e.x,a=e.y,l=e.z,d=Math.cos(r),m=Math.sin(r),g=Math.cos(a),_=Math.sin(a),M=Math.cos(l),u=Math.sin(l);if(e.order==="XYZ"){const f=d*M,p=d*u,y=m*M,E=m*u;t[0]=g*M,t[4]=-g*u,t[8]=_,t[1]=p+y*_,t[5]=f-E*_,t[9]=-m*g,t[2]=E-f*_,t[6]=y+p*_,t[10]=d*g}else if(e.order==="YXZ"){const f=g*M,p=g*u,y=_*M,E=_*u;t[0]=f+E*m,t[4]=y*m-p,t[8]=d*_,t[1]=d*u,t[5]=d*M,t[9]=-m,t[2]=p*m-y,t[6]=E+f*m,t[10]=d*g}else if(e.order==="ZXY"){const f=g*M,p=g*u,y=_*M,E=_*u;t[0]=f-E*m,t[4]=-d*u,t[8]=y+p*m,t[1]=p+y*m,t[5]=d*M,t[9]=E-f*m,t[2]=-d*_,t[6]=m,t[10]=d*g}else if(e.order==="ZYX"){const f=d*M,p=d*u,y=m*M,E=m*u;t[0]=g*M,t[4]=y*_-p,t[8]=f*_+E,t[1]=g*u,t[5]=E*_+f,t[9]=p*_-y,t[2]=-_,t[6]=m*g,t[10]=d*g}else if(e.order==="YZX"){const f=d*g,p=d*_,y=m*g,E=m*_;t[0]=g*M,t[4]=E-f*u,t[8]=y*u+p,t[1]=u,t[5]=d*M,t[9]=-m*M,t[2]=-_*M,t[6]=p*u+y,t[10]=f-E*u}else if(e.order==="XZY"){const f=d*g,p=d*_,y=m*g,E=m*_;t[0]=g*M,t[4]=-u,t[8]=_*M,t[1]=f*u+E,t[5]=d*M,t[9]=p*u-y,t[2]=y*u-p,t[6]=m*M,t[10]=E*u+f}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(Bv,e,kv)}lookAt(e,t,r){const a=this.elements;return Xn.subVectors(e,t),Xn.lengthSq()===0&&(Xn.z=1),Xn.normalize(),Tr.crossVectors(r,Xn),Tr.lengthSq()===0&&(Math.abs(r.z)===1?Xn.x+=1e-4:Xn.z+=1e-4,Xn.normalize(),Tr.crossVectors(r,Xn)),Tr.normalize(),hl.crossVectors(Xn,Tr),a[0]=Tr.x,a[4]=hl.x,a[8]=Xn.x,a[1]=Tr.y,a[5]=hl.y,a[9]=Xn.y,a[2]=Tr.z,a[6]=hl.z,a[10]=Xn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const r=e.elements,a=t.elements,l=this.elements,d=r[0],m=r[4],g=r[8],_=r[12],M=r[1],u=r[5],f=r[9],p=r[13],y=r[2],E=r[6],S=r[10],v=r[14],A=r[3],P=r[7],L=r[11],z=r[15],D=a[0],F=a[4],R=a[8],I=a[12],W=a[1],O=a[5],j=a[9],re=a[13],ae=a[2],X=a[6],Z=a[10],q=a[14],G=a[3],J=a[7],ie=a[11],U=a[15];return l[0]=d*D+m*W+g*ae+_*G,l[4]=d*F+m*O+g*X+_*J,l[8]=d*R+m*j+g*Z+_*ie,l[12]=d*I+m*re+g*q+_*U,l[1]=M*D+u*W+f*ae+p*G,l[5]=M*F+u*O+f*X+p*J,l[9]=M*R+u*j+f*Z+p*ie,l[13]=M*I+u*re+f*q+p*U,l[2]=y*D+E*W+S*ae+v*G,l[6]=y*F+E*O+S*X+v*J,l[10]=y*R+E*j+S*Z+v*ie,l[14]=y*I+E*re+S*q+v*U,l[3]=A*D+P*W+L*ae+z*G,l[7]=A*F+P*O+L*X+z*J,l[11]=A*R+P*j+L*Z+z*ie,l[15]=A*I+P*re+L*q+z*U,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],r=e[4],a=e[8],l=e[12],d=e[1],m=e[5],g=e[9],_=e[13],M=e[2],u=e[6],f=e[10],p=e[14],y=e[3],E=e[7],S=e[11],v=e[15],A=g*p-_*f,P=m*p-_*u,L=m*f-g*u,z=d*p-_*M,D=d*f-g*M,F=d*u-m*M;return t*(E*A-S*P+v*L)-r*(y*A-S*z+v*D)+a*(y*P-E*z+v*F)-l*(y*L-E*D+S*F)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,r){const a=this.elements;return e.isVector3?(a[12]=e.x,a[13]=e.y,a[14]=e.z):(a[12]=e,a[13]=t,a[14]=r),this}invert(){const e=this.elements,t=e[0],r=e[1],a=e[2],l=e[3],d=e[4],m=e[5],g=e[6],_=e[7],M=e[8],u=e[9],f=e[10],p=e[11],y=e[12],E=e[13],S=e[14],v=e[15],A=t*m-r*d,P=t*g-a*d,L=t*_-l*d,z=r*g-a*m,D=r*_-l*m,F=a*_-l*g,R=M*E-u*y,I=M*S-f*y,W=M*v-p*y,O=u*S-f*E,j=u*v-p*E,re=f*v-p*S,ae=A*re-P*j+L*O+z*W-D*I+F*R;if(ae===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const X=1/ae;return e[0]=(m*re-g*j+_*O)*X,e[1]=(a*j-r*re-l*O)*X,e[2]=(E*F-S*D+v*z)*X,e[3]=(f*D-u*F-p*z)*X,e[4]=(g*W-d*re-_*I)*X,e[5]=(t*re-a*W+l*I)*X,e[6]=(S*L-y*F-v*P)*X,e[7]=(M*F-f*L+p*P)*X,e[8]=(d*j-m*W+_*R)*X,e[9]=(r*W-t*j-l*R)*X,e[10]=(y*D-E*L+v*A)*X,e[11]=(u*L-M*D-p*A)*X,e[12]=(m*I-d*O-g*R)*X,e[13]=(t*O-r*I+a*R)*X,e[14]=(E*P-y*z-S*A)*X,e[15]=(M*z-u*P+f*A)*X,this}scale(e){const t=this.elements,r=e.x,a=e.y,l=e.z;return t[0]*=r,t[4]*=a,t[8]*=l,t[1]*=r,t[5]*=a,t[9]*=l,t[2]*=r,t[6]*=a,t[10]*=l,t[3]*=r,t[7]*=a,t[11]*=l,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],r=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],a=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,r,a))}makeTranslation(e,t,r){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,r,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),r=Math.sin(e);return this.set(1,0,0,0,0,t,-r,0,0,r,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),r=Math.sin(e);return this.set(t,0,r,0,0,1,0,0,-r,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),r=Math.sin(e);return this.set(t,-r,0,0,r,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const r=Math.cos(t),a=Math.sin(t),l=1-r,d=e.x,m=e.y,g=e.z,_=l*d,M=l*m;return this.set(_*d+r,_*m-a*g,_*g+a*m,0,_*m+a*g,M*m+r,M*g-a*d,0,_*g-a*m,M*g+a*d,l*g*g+r,0,0,0,0,1),this}makeScale(e,t,r){return this.set(e,0,0,0,0,t,0,0,0,0,r,0,0,0,0,1),this}makeShear(e,t,r,a,l,d){return this.set(1,r,l,0,e,1,d,0,t,a,1,0,0,0,0,1),this}compose(e,t,r){const a=this.elements,l=t._x,d=t._y,m=t._z,g=t._w,_=l+l,M=d+d,u=m+m,f=l*_,p=l*M,y=l*u,E=d*M,S=d*u,v=m*u,A=g*_,P=g*M,L=g*u,z=r.x,D=r.y,F=r.z;return a[0]=(1-(E+v))*z,a[1]=(p+L)*z,a[2]=(y-P)*z,a[3]=0,a[4]=(p-L)*D,a[5]=(1-(f+v))*D,a[6]=(S+A)*D,a[7]=0,a[8]=(y+P)*F,a[9]=(S-A)*F,a[10]=(1-(f+E))*F,a[11]=0,a[12]=e.x,a[13]=e.y,a[14]=e.z,a[15]=1,this}decompose(e,t,r){const a=this.elements;e.x=a[12],e.y=a[13],e.z=a[14];const l=this.determinant();if(l===0)return r.set(1,1,1),t.identity(),this;let d=Fs.set(a[0],a[1],a[2]).length();const m=Fs.set(a[4],a[5],a[6]).length(),g=Fs.set(a[8],a[9],a[10]).length();l<0&&(d=-d),hi.copy(this);const _=1/d,M=1/m,u=1/g;return hi.elements[0]*=_,hi.elements[1]*=_,hi.elements[2]*=_,hi.elements[4]*=M,hi.elements[5]*=M,hi.elements[6]*=M,hi.elements[8]*=u,hi.elements[9]*=u,hi.elements[10]*=u,t.setFromRotationMatrix(hi),r.x=d,r.y=m,r.z=g,this}makePerspective(e,t,r,a,l,d,m=Li,g=!1){const _=this.elements,M=2*l/(t-e),u=2*l/(r-a),f=(t+e)/(t-e),p=(r+a)/(r-a);let y,E;if(g)y=l/(d-l),E=d*l/(d-l);else if(m===Li)y=-(d+l)/(d-l),E=-2*d*l/(d-l);else if(m===$l)y=-d/(d-l),E=-d*l/(d-l);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+m);return _[0]=M,_[4]=0,_[8]=f,_[12]=0,_[1]=0,_[5]=u,_[9]=p,_[13]=0,_[2]=0,_[6]=0,_[10]=y,_[14]=E,_[3]=0,_[7]=0,_[11]=-1,_[15]=0,this}makeOrthographic(e,t,r,a,l,d,m=Li,g=!1){const _=this.elements,M=2/(t-e),u=2/(r-a),f=-(t+e)/(t-e),p=-(r+a)/(r-a);let y,E;if(g)y=1/(d-l),E=d/(d-l);else if(m===Li)y=-2/(d-l),E=-(d+l)/(d-l);else if(m===$l)y=-1/(d-l),E=-l/(d-l);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+m);return _[0]=M,_[4]=0,_[8]=0,_[12]=f,_[1]=0,_[5]=u,_[9]=0,_[13]=p,_[2]=0,_[6]=0,_[10]=y,_[14]=E,_[3]=0,_[7]=0,_[11]=0,_[15]=1,this}equals(e){const t=this.elements,r=e.elements;for(let a=0;a<16;a++)if(t[a]!==r[a])return!1;return!0}fromArray(e,t=0){for(let r=0;r<16;r++)this.elements[r]=e[r+t];return this}toArray(e=[],t=0){const r=this.elements;return e[t]=r[0],e[t+1]=r[1],e[t+2]=r[2],e[t+3]=r[3],e[t+4]=r[4],e[t+5]=r[5],e[t+6]=r[6],e[t+7]=r[7],e[t+8]=r[8],e[t+9]=r[9],e[t+10]=r[10],e[t+11]=r[11],e[t+12]=r[12],e[t+13]=r[13],e[t+14]=r[14],e[t+15]=r[15],e}};Ql.prototype.isMatrix4=!0;let rn=Ql;const Fs=new oe,hi=new rn,Bv=new oe(0,0,0),kv=new oe(1,1,1),Tr=new oe,hl=new oe,Xn=new oe,_m=new rn,gm=new io;class as{constructor(e=0,t=0,r=0,a=as.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=r,this._order=a}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,r,a=this._order){return this._x=e,this._y=t,this._z=r,this._order=a,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,r=!0){const a=e.elements,l=a[0],d=a[4],m=a[8],g=a[1],_=a[5],M=a[9],u=a[2],f=a[6],p=a[10];switch(t){case"XYZ":this._y=Math.asin(vt(m,-1,1)),Math.abs(m)<.9999999?(this._x=Math.atan2(-M,p),this._z=Math.atan2(-d,l)):(this._x=Math.atan2(f,_),this._z=0);break;case"YXZ":this._x=Math.asin(-vt(M,-1,1)),Math.abs(M)<.9999999?(this._y=Math.atan2(m,p),this._z=Math.atan2(g,_)):(this._y=Math.atan2(-u,l),this._z=0);break;case"ZXY":this._x=Math.asin(vt(f,-1,1)),Math.abs(f)<.9999999?(this._y=Math.atan2(-u,p),this._z=Math.atan2(-d,_)):(this._y=0,this._z=Math.atan2(g,l));break;case"ZYX":this._y=Math.asin(-vt(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(f,p),this._z=Math.atan2(g,l)):(this._x=0,this._z=Math.atan2(-d,_));break;case"YZX":this._z=Math.asin(vt(g,-1,1)),Math.abs(g)<.9999999?(this._x=Math.atan2(-M,_),this._y=Math.atan2(-u,l)):(this._x=0,this._y=Math.atan2(m,p));break;case"XZY":this._z=Math.asin(-vt(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(f,_),this._y=Math.atan2(m,l)):(this._x=Math.atan2(-M,p),this._y=0);break;default:tt("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,r===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,r){return _m.makeRotationFromQuaternion(e),this.setFromRotationMatrix(_m,t,r)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return gm.setFromEuler(this),this.setFromQuaternion(gm,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}as.DEFAULT_ORDER="XYZ";class w_{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let zv=0;const vm=new oe,Os=new io,Yi=new rn,pl=new oe,Wo=new oe,Hv=new oe,Vv=new io,xm=new oe(1,0,0),Sm=new oe(0,1,0),ym=new oe(0,0,1),Mm={type:"added"},Gv={type:"removed"},Bs={type:"childadded",child:null},jc={type:"childremoved",child:null};class zn extends ls{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:zv++}),this.uuid=no(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=zn.DEFAULT_UP.clone();const e=new oe,t=new as,r=new io,a=new oe(1,1,1);function l(){r.setFromEuler(t,!1)}function d(){t.setFromQuaternion(r,void 0,!1)}t._onChange(l),r._onChange(d),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:r},scale:{configurable:!0,enumerable:!0,value:a},modelViewMatrix:{value:new rn},normalMatrix:{value:new lt}}),this.matrix=new rn,this.matrixWorld=new rn,this.matrixAutoUpdate=zn.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=zn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new w_,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Os.setFromAxisAngle(e,t),this.quaternion.multiply(Os),this}rotateOnWorldAxis(e,t){return Os.setFromAxisAngle(e,t),this.quaternion.premultiply(Os),this}rotateX(e){return this.rotateOnAxis(xm,e)}rotateY(e){return this.rotateOnAxis(Sm,e)}rotateZ(e){return this.rotateOnAxis(ym,e)}translateOnAxis(e,t){return vm.copy(e).applyQuaternion(this.quaternion),this.position.add(vm.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(xm,e)}translateY(e){return this.translateOnAxis(Sm,e)}translateZ(e){return this.translateOnAxis(ym,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Yi.copy(this.matrixWorld).invert())}lookAt(e,t,r){e.isVector3?pl.copy(e):pl.set(e,t,r);const a=this.parent;this.updateWorldMatrix(!0,!1),Wo.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Yi.lookAt(Wo,pl,this.up):Yi.lookAt(pl,Wo,this.up),this.quaternion.setFromRotationMatrix(Yi),a&&(Yi.extractRotation(a.matrixWorld),Os.setFromRotationMatrix(Yi),this.quaternion.premultiply(Os.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(Mt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Mm),Bs.child=e,this.dispatchEvent(Bs),Bs.child=null):Mt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let r=0;r<arguments.length;r++)this.remove(arguments[r]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Gv),jc.child=e,this.dispatchEvent(jc),jc.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Yi.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Yi.multiply(e.parent.matrixWorld)),e.applyMatrix4(Yi),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Mm),Bs.child=e,this.dispatchEvent(Bs),Bs.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let r=0,a=this.children.length;r<a;r++){const d=this.children[r].getObjectByProperty(e,t);if(d!==void 0)return d}}getObjectsByProperty(e,t,r=[]){this[e]===t&&r.push(this);const a=this.children;for(let l=0,d=a.length;l<d;l++)a[l].getObjectsByProperty(e,t,r);return r}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Wo,e,Hv),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Wo,Vv,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let r=0,a=t.length;r<a;r++)t[r].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let r=0,a=t.length;r<a;r++)t[r].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);const e=this.pivot;if(e!==null){const t=e.x,r=e.y,a=e.z,l=this.matrix.elements;l[12]+=t-l[0]*t-l[4]*r-l[8]*a,l[13]+=r-l[1]*t-l[5]*r-l[9]*a,l[14]+=a-l[2]*t-l[6]*r-l[10]*a}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let r=0,a=t.length;r<a;r++)t[r].updateMatrixWorld(e)}updateWorldMatrix(e,t){const r=this.parent;if(e===!0&&r!==null&&r.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const a=this.children;for(let l=0,d=a.length;l<d;l++)a[l].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",r={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},r.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const a={};a.uuid=this.uuid,a.type=this.type,this.name!==""&&(a.name=this.name),this.castShadow===!0&&(a.castShadow=!0),this.receiveShadow===!0&&(a.receiveShadow=!0),this.visible===!1&&(a.visible=!1),this.frustumCulled===!1&&(a.frustumCulled=!1),this.renderOrder!==0&&(a.renderOrder=this.renderOrder),this.static!==!1&&(a.static=this.static),Object.keys(this.userData).length>0&&(a.userData=this.userData),a.layers=this.layers.mask,a.matrix=this.matrix.toArray(),a.up=this.up.toArray(),this.pivot!==null&&(a.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(a.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(a.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(a.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(a.type="InstancedMesh",a.count=this.count,a.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(a.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(a.type="BatchedMesh",a.perObjectFrustumCulled=this.perObjectFrustumCulled,a.sortObjects=this.sortObjects,a.drawRanges=this._drawRanges,a.reservedRanges=this._reservedRanges,a.geometryInfo=this._geometryInfo.map(m=>({...m,boundingBox:m.boundingBox?m.boundingBox.toJSON():void 0,boundingSphere:m.boundingSphere?m.boundingSphere.toJSON():void 0})),a.instanceInfo=this._instanceInfo.map(m=>({...m})),a.availableInstanceIds=this._availableInstanceIds.slice(),a.availableGeometryIds=this._availableGeometryIds.slice(),a.nextIndexStart=this._nextIndexStart,a.nextVertexStart=this._nextVertexStart,a.geometryCount=this._geometryCount,a.maxInstanceCount=this._maxInstanceCount,a.maxVertexCount=this._maxVertexCount,a.maxIndexCount=this._maxIndexCount,a.geometryInitialized=this._geometryInitialized,a.matricesTexture=this._matricesTexture.toJSON(e),a.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(a.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(a.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(a.boundingBox=this.boundingBox.toJSON()));function l(m,g){return m[g.uuid]===void 0&&(m[g.uuid]=g.toJSON(e)),g.uuid}if(this.isScene)this.background&&(this.background.isColor?a.background=this.background.toJSON():this.background.isTexture&&(a.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(a.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){a.geometry=l(e.geometries,this.geometry);const m=this.geometry.parameters;if(m!==void 0&&m.shapes!==void 0){const g=m.shapes;if(Array.isArray(g))for(let _=0,M=g.length;_<M;_++){const u=g[_];l(e.shapes,u)}else l(e.shapes,g)}}if(this.isSkinnedMesh&&(a.bindMode=this.bindMode,a.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(l(e.skeletons,this.skeleton),a.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const m=[];for(let g=0,_=this.material.length;g<_;g++)m.push(l(e.materials,this.material[g]));a.material=m}else a.material=l(e.materials,this.material);if(this.children.length>0){a.children=[];for(let m=0;m<this.children.length;m++)a.children.push(this.children[m].toJSON(e).object)}if(this.animations.length>0){a.animations=[];for(let m=0;m<this.animations.length;m++){const g=this.animations[m];a.animations.push(l(e.animations,g))}}if(t){const m=d(e.geometries),g=d(e.materials),_=d(e.textures),M=d(e.images),u=d(e.shapes),f=d(e.skeletons),p=d(e.animations),y=d(e.nodes);m.length>0&&(r.geometries=m),g.length>0&&(r.materials=g),_.length>0&&(r.textures=_),M.length>0&&(r.images=M),u.length>0&&(r.shapes=u),f.length>0&&(r.skeletons=f),p.length>0&&(r.animations=p),y.length>0&&(r.nodes=y)}return r.object=a,r;function d(m){const g=[];for(const _ in m){const M=m[_];delete M.metadata,g.push(M)}return g}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let r=0;r<e.children.length;r++){const a=e.children[r];this.add(a.clone())}return this}}zn.DEFAULT_UP=new oe(0,1,0);zn.DEFAULT_MATRIX_AUTO_UPDATE=!0;zn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;class ml extends zn{constructor(){super(),this.isGroup=!0,this.type="Group"}}const Wv={type:"move"};class Kc{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new ml,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new ml,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new oe,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new oe),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new ml,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new oe,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new oe,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const r of e.hand.values())this._getHandJoint(t,r)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,r){let a=null,l=null,d=null;const m=this._targetRay,g=this._grip,_=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(_&&e.hand){d=!0;for(const E of e.hand.values()){const S=t.getJointPose(E,r),v=this._getHandJoint(_,E);S!==null&&(v.matrix.fromArray(S.transform.matrix),v.matrix.decompose(v.position,v.rotation,v.scale),v.matrixWorldNeedsUpdate=!0,v.jointRadius=S.radius),v.visible=S!==null}const M=_.joints["index-finger-tip"],u=_.joints["thumb-tip"],f=M.position.distanceTo(u.position),p=.02,y=.005;_.inputState.pinching&&f>p+y?(_.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!_.inputState.pinching&&f<=p-y&&(_.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else g!==null&&e.gripSpace&&(l=t.getPose(e.gripSpace,r),l!==null&&(g.matrix.fromArray(l.transform.matrix),g.matrix.decompose(g.position,g.rotation,g.scale),g.matrixWorldNeedsUpdate=!0,l.linearVelocity?(g.hasLinearVelocity=!0,g.linearVelocity.copy(l.linearVelocity)):g.hasLinearVelocity=!1,l.angularVelocity?(g.hasAngularVelocity=!0,g.angularVelocity.copy(l.angularVelocity)):g.hasAngularVelocity=!1,g.eventsEnabled&&g.dispatchEvent({type:"gripUpdated",data:e,target:this})));m!==null&&(a=t.getPose(e.targetRaySpace,r),a===null&&l!==null&&(a=l),a!==null&&(m.matrix.fromArray(a.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,a.linearVelocity?(m.hasLinearVelocity=!0,m.linearVelocity.copy(a.linearVelocity)):m.hasLinearVelocity=!1,a.angularVelocity?(m.hasAngularVelocity=!0,m.angularVelocity.copy(a.angularVelocity)):m.hasAngularVelocity=!1,this.dispatchEvent(Wv)))}return m!==null&&(m.visible=a!==null),g!==null&&(g.visible=l!==null),_!==null&&(_.visible=d!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const r=new ml;r.matrixAutoUpdate=!1,r.visible=!1,e.joints[t.jointName]=r,e.add(r)}return e.joints[t.jointName]}}const A_={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},wr={h:0,s:0,l:0},_l={h:0,s:0,l:0};function $c(s,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?s+(e-s)*6*t:t<1/2?e:t<2/3?s+(e-s)*6*(2/3-t):s}class At{constructor(e,t,r){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,r)}set(e,t,r){if(t===void 0&&r===void 0){const a=e;a&&a.isColor?this.copy(a):typeof a=="number"?this.setHex(a):typeof a=="string"&&this.setStyle(a)}else this.setRGB(e,t,r);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=ti){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,xt.colorSpaceToWorking(this,t),this}setRGB(e,t,r,a=xt.workingColorSpace){return this.r=e,this.g=t,this.b=r,xt.colorSpaceToWorking(this,a),this}setHSL(e,t,r,a=xt.workingColorSpace){if(e=Ed(e,1),t=vt(t,0,1),r=vt(r,0,1),t===0)this.r=this.g=this.b=r;else{const l=r<=.5?r*(1+t):r+t-r*t,d=2*r-l;this.r=$c(d,l,e+1/3),this.g=$c(d,l,e),this.b=$c(d,l,e-1/3)}return xt.colorSpaceToWorking(this,a),this}setStyle(e,t=ti){function r(l){l!==void 0&&parseFloat(l)<1&&tt("Color: Alpha component of "+e+" will be ignored.")}let a;if(a=/^(\w+)\(([^\)]*)\)/.exec(e)){let l;const d=a[1],m=a[2];switch(d){case"rgb":case"rgba":if(l=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(m))return r(l[4]),this.setRGB(Math.min(255,parseInt(l[1],10))/255,Math.min(255,parseInt(l[2],10))/255,Math.min(255,parseInt(l[3],10))/255,t);if(l=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(m))return r(l[4]),this.setRGB(Math.min(100,parseInt(l[1],10))/100,Math.min(100,parseInt(l[2],10))/100,Math.min(100,parseInt(l[3],10))/100,t);break;case"hsl":case"hsla":if(l=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(m))return r(l[4]),this.setHSL(parseFloat(l[1])/360,parseFloat(l[2])/100,parseFloat(l[3])/100,t);break;default:tt("Color: Unknown color model "+e)}}else if(a=/^\#([A-Fa-f\d]+)$/.exec(e)){const l=a[1],d=l.length;if(d===3)return this.setRGB(parseInt(l.charAt(0),16)/15,parseInt(l.charAt(1),16)/15,parseInt(l.charAt(2),16)/15,t);if(d===6)return this.setHex(parseInt(l,16),t);tt("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=ti){const r=A_[e.toLowerCase()];return r!==void 0?this.setHex(r,t):tt("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=er(e.r),this.g=er(e.g),this.b=er(e.b),this}copyLinearToSRGB(e){return this.r=$s(e.r),this.g=$s(e.g),this.b=$s(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=ti){return xt.workingToColorSpace(En.copy(this),e),Math.round(vt(En.r*255,0,255))*65536+Math.round(vt(En.g*255,0,255))*256+Math.round(vt(En.b*255,0,255))}getHexString(e=ti){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=xt.workingColorSpace){xt.workingToColorSpace(En.copy(this),t);const r=En.r,a=En.g,l=En.b,d=Math.max(r,a,l),m=Math.min(r,a,l);let g,_;const M=(m+d)/2;if(m===d)g=0,_=0;else{const u=d-m;switch(_=M<=.5?u/(d+m):u/(2-d-m),d){case r:g=(a-l)/u+(a<l?6:0);break;case a:g=(l-r)/u+2;break;case l:g=(r-a)/u+4;break}g/=6}return e.h=g,e.s=_,e.l=M,e}getRGB(e,t=xt.workingColorSpace){return xt.workingToColorSpace(En.copy(this),t),e.r=En.r,e.g=En.g,e.b=En.b,e}getStyle(e=ti){xt.workingToColorSpace(En.copy(this),e);const t=En.r,r=En.g,a=En.b;return e!==ti?`color(${e} ${t.toFixed(3)} ${r.toFixed(3)} ${a.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(r*255)},${Math.round(a*255)})`}offsetHSL(e,t,r){return this.getHSL(wr),this.setHSL(wr.h+e,wr.s+t,wr.l+r)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,r){return this.r=e.r+(t.r-e.r)*r,this.g=e.g+(t.g-e.g)*r,this.b=e.b+(t.b-e.b)*r,this}lerpHSL(e,t){this.getHSL(wr),e.getHSL(_l);const r=ta(wr.h,_l.h,t),a=ta(wr.s,_l.s,t),l=ta(wr.l,_l.l,t);return this.setHSL(r,a,l),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,r=this.g,a=this.b,l=e.elements;return this.r=l[0]*t+l[3]*r+l[6]*a,this.g=l[1]*t+l[4]*r+l[7]*a,this.b=l[2]*t+l[5]*r+l[8]*a,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const En=new At;At.NAMES=A_;class wd{constructor(e,t=25e-5){this.isFogExp2=!0,this.name="",this.color=new At(e),this.density=t}clone(){return new wd(this.color,this.density)}toJSON(){return{type:"FogExp2",name:this.name,color:this.color.getHex(),density:this.density}}}class Xv extends zn{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new as,this.environmentIntensity=1,this.environmentRotation=new as,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}const pi=new oe,qi=new oe,Zc=new oe,ji=new oe,ks=new oe,zs=new oe,Em=new oe,Qc=new oe,Jc=new oe,ef=new oe,tf=new Jt,nf=new Jt,rf=new Jt;class _i{constructor(e=new oe,t=new oe,r=new oe){this.a=e,this.b=t,this.c=r}static getNormal(e,t,r,a){a.subVectors(r,t),pi.subVectors(e,t),a.cross(pi);const l=a.lengthSq();return l>0?a.multiplyScalar(1/Math.sqrt(l)):a.set(0,0,0)}static getBarycoord(e,t,r,a,l){pi.subVectors(a,t),qi.subVectors(r,t),Zc.subVectors(e,t);const d=pi.dot(pi),m=pi.dot(qi),g=pi.dot(Zc),_=qi.dot(qi),M=qi.dot(Zc),u=d*_-m*m;if(u===0)return l.set(0,0,0),null;const f=1/u,p=(_*g-m*M)*f,y=(d*M-m*g)*f;return l.set(1-p-y,y,p)}static containsPoint(e,t,r,a){return this.getBarycoord(e,t,r,a,ji)===null?!1:ji.x>=0&&ji.y>=0&&ji.x+ji.y<=1}static getInterpolation(e,t,r,a,l,d,m,g){return this.getBarycoord(e,t,r,a,ji)===null?(g.x=0,g.y=0,"z"in g&&(g.z=0),"w"in g&&(g.w=0),null):(g.setScalar(0),g.addScaledVector(l,ji.x),g.addScaledVector(d,ji.y),g.addScaledVector(m,ji.z),g)}static getInterpolatedAttribute(e,t,r,a,l,d){return tf.setScalar(0),nf.setScalar(0),rf.setScalar(0),tf.fromBufferAttribute(e,t),nf.fromBufferAttribute(e,r),rf.fromBufferAttribute(e,a),d.setScalar(0),d.addScaledVector(tf,l.x),d.addScaledVector(nf,l.y),d.addScaledVector(rf,l.z),d}static isFrontFacing(e,t,r,a){return pi.subVectors(r,t),qi.subVectors(e,t),pi.cross(qi).dot(a)<0}set(e,t,r){return this.a.copy(e),this.b.copy(t),this.c.copy(r),this}setFromPointsAndIndices(e,t,r,a){return this.a.copy(e[t]),this.b.copy(e[r]),this.c.copy(e[a]),this}setFromAttributeAndIndices(e,t,r,a){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,r),this.c.fromBufferAttribute(e,a),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return pi.subVectors(this.c,this.b),qi.subVectors(this.a,this.b),pi.cross(qi).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return _i.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return _i.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,r,a,l){return _i.getInterpolation(e,this.a,this.b,this.c,t,r,a,l)}containsPoint(e){return _i.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return _i.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const r=this.a,a=this.b,l=this.c;let d,m;ks.subVectors(a,r),zs.subVectors(l,r),Qc.subVectors(e,r);const g=ks.dot(Qc),_=zs.dot(Qc);if(g<=0&&_<=0)return t.copy(r);Jc.subVectors(e,a);const M=ks.dot(Jc),u=zs.dot(Jc);if(M>=0&&u<=M)return t.copy(a);const f=g*u-M*_;if(f<=0&&g>=0&&M<=0)return d=g/(g-M),t.copy(r).addScaledVector(ks,d);ef.subVectors(e,l);const p=ks.dot(ef),y=zs.dot(ef);if(y>=0&&p<=y)return t.copy(l);const E=p*_-g*y;if(E<=0&&_>=0&&y<=0)return m=_/(_-y),t.copy(r).addScaledVector(zs,m);const S=M*y-p*u;if(S<=0&&u-M>=0&&p-y>=0)return Em.subVectors(l,a),m=(u-M)/(u-M+(p-y)),t.copy(a).addScaledVector(Em,m);const v=1/(S+E+f);return d=E*v,m=f*v,t.copy(r).addScaledVector(ks,d).addScaledVector(zs,m)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}class oa{constructor(e=new oe(1/0,1/0,1/0),t=new oe(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,r=e.length;t<r;t+=3)this.expandByPoint(mi.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,r=e.count;t<r;t++)this.expandByPoint(mi.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,r=e.length;t<r;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const r=mi.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(r),this.max.copy(e).add(r),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const r=e.geometry;if(r!==void 0){const l=r.getAttribute("position");if(t===!0&&l!==void 0&&e.isInstancedMesh!==!0)for(let d=0,m=l.count;d<m;d++)e.isMesh===!0?e.getVertexPosition(d,mi):mi.fromBufferAttribute(l,d),mi.applyMatrix4(e.matrixWorld),this.expandByPoint(mi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),gl.copy(e.boundingBox)):(r.boundingBox===null&&r.computeBoundingBox(),gl.copy(r.boundingBox)),gl.applyMatrix4(e.matrixWorld),this.union(gl)}const a=e.children;for(let l=0,d=a.length;l<d;l++)this.expandByObject(a[l],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,mi),mi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,r;return e.normal.x>0?(t=e.normal.x*this.min.x,r=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,r=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,r+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,r+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,r+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,r+=e.normal.z*this.min.z),t<=-e.constant&&r>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Xo),vl.subVectors(this.max,Xo),Hs.subVectors(e.a,Xo),Vs.subVectors(e.b,Xo),Gs.subVectors(e.c,Xo),Ar.subVectors(Vs,Hs),Rr.subVectors(Gs,Vs),jr.subVectors(Hs,Gs);let t=[0,-Ar.z,Ar.y,0,-Rr.z,Rr.y,0,-jr.z,jr.y,Ar.z,0,-Ar.x,Rr.z,0,-Rr.x,jr.z,0,-jr.x,-Ar.y,Ar.x,0,-Rr.y,Rr.x,0,-jr.y,jr.x,0];return!sf(t,Hs,Vs,Gs,vl)||(t=[1,0,0,0,1,0,0,0,1],!sf(t,Hs,Vs,Gs,vl))?!1:(xl.crossVectors(Ar,Rr),t=[xl.x,xl.y,xl.z],sf(t,Hs,Vs,Gs,vl))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,mi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(mi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Ki[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Ki[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Ki[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Ki[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Ki[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Ki[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Ki[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Ki[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Ki),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Ki=[new oe,new oe,new oe,new oe,new oe,new oe,new oe,new oe],mi=new oe,gl=new oa,Hs=new oe,Vs=new oe,Gs=new oe,Ar=new oe,Rr=new oe,jr=new oe,Xo=new oe,vl=new oe,xl=new oe,Kr=new oe;function sf(s,e,t,r,a){for(let l=0,d=s.length-3;l<=d;l+=3){Kr.fromArray(s,l);const m=a.x*Math.abs(Kr.x)+a.y*Math.abs(Kr.y)+a.z*Math.abs(Kr.z),g=e.dot(Kr),_=t.dot(Kr),M=r.dot(Kr);if(Math.max(-Math.max(g,_,M),Math.min(g,_,M))>m)return!1}return!0}const nn=new oe,Sl=new It;let Yv=0;class Zt extends ls{constructor(e,t,r=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Yv++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=r,this.usage=lm,this.updateRanges=[],this.gpuType=Pi,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,r){e*=this.itemSize,r*=t.itemSize;for(let a=0,l=this.itemSize;a<l;a++)this.array[e+a]=t.array[r+a];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,r=this.count;t<r;t++)Sl.fromBufferAttribute(this,t),Sl.applyMatrix3(e),this.setXY(t,Sl.x,Sl.y);else if(this.itemSize===3)for(let t=0,r=this.count;t<r;t++)nn.fromBufferAttribute(this,t),nn.applyMatrix3(e),this.setXYZ(t,nn.x,nn.y,nn.z);return this}applyMatrix4(e){for(let t=0,r=this.count;t<r;t++)nn.fromBufferAttribute(this,t),nn.applyMatrix4(e),this.setXYZ(t,nn.x,nn.y,nn.z);return this}applyNormalMatrix(e){for(let t=0,r=this.count;t<r;t++)nn.fromBufferAttribute(this,t),nn.applyNormalMatrix(e),this.setXYZ(t,nn.x,nn.y,nn.z);return this}transformDirection(e){for(let t=0,r=this.count;t<r;t++)nn.fromBufferAttribute(this,t),nn.transformDirection(e),this.setXYZ(t,nn.x,nn.y,nn.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let r=this.array[e*this.itemSize+t];return this.normalized&&(r=js(r,this.array)),r}setComponent(e,t,r){return this.normalized&&(r=Cn(r,this.array)),this.array[e*this.itemSize+t]=r,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=js(t,this.array)),t}setX(e,t){return this.normalized&&(t=Cn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=js(t,this.array)),t}setY(e,t){return this.normalized&&(t=Cn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=js(t,this.array)),t}setZ(e,t){return this.normalized&&(t=Cn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=js(t,this.array)),t}setW(e,t){return this.normalized&&(t=Cn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,r){return e*=this.itemSize,this.normalized&&(t=Cn(t,this.array),r=Cn(r,this.array)),this.array[e+0]=t,this.array[e+1]=r,this}setXYZ(e,t,r,a){return e*=this.itemSize,this.normalized&&(t=Cn(t,this.array),r=Cn(r,this.array),a=Cn(a,this.array)),this.array[e+0]=t,this.array[e+1]=r,this.array[e+2]=a,this}setXYZW(e,t,r,a,l){return e*=this.itemSize,this.normalized&&(t=Cn(t,this.array),r=Cn(r,this.array),a=Cn(a,this.array),l=Cn(l,this.array)),this.array[e+0]=t,this.array[e+1]=r,this.array[e+2]=a,this.array[e+3]=l,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==lm&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:"dispose"})}}class R_ extends Zt{constructor(e,t,r){super(new Uint16Array(e),t,r)}}class C_ extends Zt{constructor(e,t,r){super(new Uint32Array(e),t,r)}}class tr extends Zt{constructor(e,t,r){super(new Float32Array(e),t,r)}}const qv=new oa,Yo=new oe,of=new oe;class eu{constructor(e=new oe,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const r=this.center;t!==void 0?r.copy(t):qv.setFromPoints(e).getCenter(r);let a=0;for(let l=0,d=e.length;l<d;l++)a=Math.max(a,r.distanceToSquared(e[l]));return this.radius=Math.sqrt(a),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const r=this.center.distanceToSquared(e);return t.copy(e),r>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Yo.subVectors(e,this.center);const t=Yo.lengthSq();if(t>this.radius*this.radius){const r=Math.sqrt(t),a=(r-this.radius)*.5;this.center.addScaledVector(Yo,a/r),this.radius+=a}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(of.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Yo.copy(e.center).add(of)),this.expandByPoint(Yo.copy(e.center).sub(of))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}let jv=0;const ei=new rn,af=new zn,Ws=new oe,Yn=new oa,qo=new oa,dn=new oe;class ri extends ls{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:jv++}),this.uuid=no(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(fv(e)?C_:R_)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,r=0){this.groups.push({start:e,count:t,materialIndex:r})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const r=this.attributes.normal;if(r!==void 0){const l=new lt().getNormalMatrix(e);r.applyNormalMatrix(l),r.needsUpdate=!0}const a=this.attributes.tangent;return a!==void 0&&(a.transformDirection(e),a.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return ei.makeRotationFromQuaternion(e),this.applyMatrix4(ei),this}rotateX(e){return ei.makeRotationX(e),this.applyMatrix4(ei),this}rotateY(e){return ei.makeRotationY(e),this.applyMatrix4(ei),this}rotateZ(e){return ei.makeRotationZ(e),this.applyMatrix4(ei),this}translate(e,t,r){return ei.makeTranslation(e,t,r),this.applyMatrix4(ei),this}scale(e,t,r){return ei.makeScale(e,t,r),this.applyMatrix4(ei),this}lookAt(e){return af.lookAt(e),af.updateMatrix(),this.applyMatrix4(af.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Ws).negate(),this.translate(Ws.x,Ws.y,Ws.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const r=[];for(let a=0,l=e.length;a<l;a++){const d=e[a];r.push(d.x,d.y,d.z||0)}this.setAttribute("position",new tr(r,3))}else{const r=Math.min(e.length,t.count);for(let a=0;a<r;a++){const l=e[a];t.setXYZ(a,l.x,l.y,l.z||0)}e.length>t.count&&tt("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new oa);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Mt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new oe(-1/0,-1/0,-1/0),new oe(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let r=0,a=t.length;r<a;r++){const l=t[r];Yn.setFromBufferAttribute(l),this.morphTargetsRelative?(dn.addVectors(this.boundingBox.min,Yn.min),this.boundingBox.expandByPoint(dn),dn.addVectors(this.boundingBox.max,Yn.max),this.boundingBox.expandByPoint(dn)):(this.boundingBox.expandByPoint(Yn.min),this.boundingBox.expandByPoint(Yn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Mt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new eu);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Mt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new oe,1/0);return}if(e){const r=this.boundingSphere.center;if(Yn.setFromBufferAttribute(e),t)for(let l=0,d=t.length;l<d;l++){const m=t[l];qo.setFromBufferAttribute(m),this.morphTargetsRelative?(dn.addVectors(Yn.min,qo.min),Yn.expandByPoint(dn),dn.addVectors(Yn.max,qo.max),Yn.expandByPoint(dn)):(Yn.expandByPoint(qo.min),Yn.expandByPoint(qo.max))}Yn.getCenter(r);let a=0;for(let l=0,d=e.count;l<d;l++)dn.fromBufferAttribute(e,l),a=Math.max(a,r.distanceToSquared(dn));if(t)for(let l=0,d=t.length;l<d;l++){const m=t[l],g=this.morphTargetsRelative;for(let _=0,M=m.count;_<M;_++)dn.fromBufferAttribute(m,_),g&&(Ws.fromBufferAttribute(e,_),dn.add(Ws)),a=Math.max(a,r.distanceToSquared(dn))}this.boundingSphere.radius=Math.sqrt(a),isNaN(this.boundingSphere.radius)&&Mt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){Mt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const r=t.position,a=t.normal,l=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Zt(new Float32Array(4*r.count),4));const d=this.getAttribute("tangent"),m=[],g=[];for(let R=0;R<r.count;R++)m[R]=new oe,g[R]=new oe;const _=new oe,M=new oe,u=new oe,f=new It,p=new It,y=new It,E=new oe,S=new oe;function v(R,I,W){_.fromBufferAttribute(r,R),M.fromBufferAttribute(r,I),u.fromBufferAttribute(r,W),f.fromBufferAttribute(l,R),p.fromBufferAttribute(l,I),y.fromBufferAttribute(l,W),M.sub(_),u.sub(_),p.sub(f),y.sub(f);const O=1/(p.x*y.y-y.x*p.y);isFinite(O)&&(E.copy(M).multiplyScalar(y.y).addScaledVector(u,-p.y).multiplyScalar(O),S.copy(u).multiplyScalar(p.x).addScaledVector(M,-y.x).multiplyScalar(O),m[R].add(E),m[I].add(E),m[W].add(E),g[R].add(S),g[I].add(S),g[W].add(S))}let A=this.groups;A.length===0&&(A=[{start:0,count:e.count}]);for(let R=0,I=A.length;R<I;++R){const W=A[R],O=W.start,j=W.count;for(let re=O,ae=O+j;re<ae;re+=3)v(e.getX(re+0),e.getX(re+1),e.getX(re+2))}const P=new oe,L=new oe,z=new oe,D=new oe;function F(R){z.fromBufferAttribute(a,R),D.copy(z);const I=m[R];P.copy(I),P.sub(z.multiplyScalar(z.dot(I))).normalize(),L.crossVectors(D,I);const O=L.dot(g[R])<0?-1:1;d.setXYZW(R,P.x,P.y,P.z,O)}for(let R=0,I=A.length;R<I;++R){const W=A[R],O=W.start,j=W.count;for(let re=O,ae=O+j;re<ae;re+=3)F(e.getX(re+0)),F(e.getX(re+1)),F(e.getX(re+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let r=this.getAttribute("normal");if(r===void 0)r=new Zt(new Float32Array(t.count*3),3),this.setAttribute("normal",r);else for(let f=0,p=r.count;f<p;f++)r.setXYZ(f,0,0,0);const a=new oe,l=new oe,d=new oe,m=new oe,g=new oe,_=new oe,M=new oe,u=new oe;if(e)for(let f=0,p=e.count;f<p;f+=3){const y=e.getX(f+0),E=e.getX(f+1),S=e.getX(f+2);a.fromBufferAttribute(t,y),l.fromBufferAttribute(t,E),d.fromBufferAttribute(t,S),M.subVectors(d,l),u.subVectors(a,l),M.cross(u),m.fromBufferAttribute(r,y),g.fromBufferAttribute(r,E),_.fromBufferAttribute(r,S),m.add(M),g.add(M),_.add(M),r.setXYZ(y,m.x,m.y,m.z),r.setXYZ(E,g.x,g.y,g.z),r.setXYZ(S,_.x,_.y,_.z)}else for(let f=0,p=t.count;f<p;f+=3)a.fromBufferAttribute(t,f+0),l.fromBufferAttribute(t,f+1),d.fromBufferAttribute(t,f+2),M.subVectors(d,l),u.subVectors(a,l),M.cross(u),r.setXYZ(f+0,M.x,M.y,M.z),r.setXYZ(f+1,M.x,M.y,M.z),r.setXYZ(f+2,M.x,M.y,M.z);this.normalizeNormals(),r.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,r=e.count;t<r;t++)dn.fromBufferAttribute(e,t),dn.normalize(),e.setXYZ(t,dn.x,dn.y,dn.z)}toNonIndexed(){function e(m,g){const _=m.array,M=m.itemSize,u=m.normalized,f=new _.constructor(g.length*M);let p=0,y=0;for(let E=0,S=g.length;E<S;E++){m.isInterleavedBufferAttribute?p=g[E]*m.data.stride+m.offset:p=g[E]*M;for(let v=0;v<M;v++)f[y++]=_[p++]}return new Zt(f,M,u)}if(this.index===null)return tt("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new ri,r=this.index.array,a=this.attributes;for(const m in a){const g=a[m],_=e(g,r);t.setAttribute(m,_)}const l=this.morphAttributes;for(const m in l){const g=[],_=l[m];for(let M=0,u=_.length;M<u;M++){const f=_[M],p=e(f,r);g.push(p)}t.morphAttributes[m]=g}t.morphTargetsRelative=this.morphTargetsRelative;const d=this.groups;for(let m=0,g=d.length;m<g;m++){const _=d[m];t.addGroup(_.start,_.count,_.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const g=this.parameters;for(const _ in g)g[_]!==void 0&&(e[_]=g[_]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const r=this.attributes;for(const g in r){const _=r[g];e.data.attributes[g]=_.toJSON(e.data)}const a={};let l=!1;for(const g in this.morphAttributes){const _=this.morphAttributes[g],M=[];for(let u=0,f=_.length;u<f;u++){const p=_[u];M.push(p.toJSON(e.data))}M.length>0&&(a[g]=M,l=!0)}l&&(e.data.morphAttributes=a,e.data.morphTargetsRelative=this.morphTargetsRelative);const d=this.groups;d.length>0&&(e.data.groups=JSON.parse(JSON.stringify(d)));const m=this.boundingSphere;return m!==null&&(e.data.boundingSphere=m.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const r=e.index;r!==null&&this.setIndex(r.clone());const a=e.attributes;for(const _ in a){const M=a[_];this.setAttribute(_,M.clone(t))}const l=e.morphAttributes;for(const _ in l){const M=[],u=l[_];for(let f=0,p=u.length;f<p;f++)M.push(u[f].clone(t));this.morphAttributes[_]=M}this.morphTargetsRelative=e.morphTargetsRelative;const d=e.groups;for(let _=0,M=d.length;_<M;_++){const u=d[_];this.addGroup(u.start,u.count,u.materialIndex)}const m=e.boundingBox;m!==null&&(this.boundingBox=m.clone());const g=e.boundingSphere;return g!==null&&(this.boundingSphere=g.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}let Kv=0;class aa extends ls{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Kv++}),this.uuid=no(),this.name="",this.type="Material",this.blending=Ks,this.side=Dr,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Mf,this.blendDst=Ef,this.blendEquation=es,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new At(0,0,0),this.blendAlpha=0,this.depthFunc=Zs,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=am,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Ns,this.stencilZFail=Ns,this.stencilZPass=Ns,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const r=e[t];if(r===void 0){tt(`Material: parameter '${t}' has value of undefined.`);continue}const a=this[t];if(a===void 0){tt(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}a&&a.isColor?a.set(r):a&&a.isVector3&&r&&r.isVector3?a.copy(r):this[t]=r}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const r={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};r.uuid=this.uuid,r.type=this.type,this.name!==""&&(r.name=this.name),this.color&&this.color.isColor&&(r.color=this.color.getHex()),this.roughness!==void 0&&(r.roughness=this.roughness),this.metalness!==void 0&&(r.metalness=this.metalness),this.sheen!==void 0&&(r.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(r.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(r.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(r.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(r.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(r.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(r.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(r.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(r.shininess=this.shininess),this.clearcoat!==void 0&&(r.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(r.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(r.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(r.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(r.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,r.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(r.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(r.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(r.dispersion=this.dispersion),this.iridescence!==void 0&&(r.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(r.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(r.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(r.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(r.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(r.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(r.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(r.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(r.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(r.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(r.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(r.lightMap=this.lightMap.toJSON(e).uuid,r.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(r.aoMap=this.aoMap.toJSON(e).uuid,r.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(r.bumpMap=this.bumpMap.toJSON(e).uuid,r.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(r.normalMap=this.normalMap.toJSON(e).uuid,r.normalMapType=this.normalMapType,r.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(r.displacementMap=this.displacementMap.toJSON(e).uuid,r.displacementScale=this.displacementScale,r.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(r.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(r.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(r.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(r.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(r.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(r.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(r.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(r.combine=this.combine)),this.envMapRotation!==void 0&&(r.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(r.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(r.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(r.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(r.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(r.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(r.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(r.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(r.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(r.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(r.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(r.size=this.size),this.shadowSide!==null&&(r.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(r.sizeAttenuation=this.sizeAttenuation),this.blending!==Ks&&(r.blending=this.blending),this.side!==Dr&&(r.side=this.side),this.vertexColors===!0&&(r.vertexColors=!0),this.opacity<1&&(r.opacity=this.opacity),this.transparent===!0&&(r.transparent=!0),this.blendSrc!==Mf&&(r.blendSrc=this.blendSrc),this.blendDst!==Ef&&(r.blendDst=this.blendDst),this.blendEquation!==es&&(r.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(r.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(r.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(r.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(r.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(r.blendAlpha=this.blendAlpha),this.depthFunc!==Zs&&(r.depthFunc=this.depthFunc),this.depthTest===!1&&(r.depthTest=this.depthTest),this.depthWrite===!1&&(r.depthWrite=this.depthWrite),this.colorWrite===!1&&(r.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(r.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==am&&(r.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(r.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(r.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Ns&&(r.stencilFail=this.stencilFail),this.stencilZFail!==Ns&&(r.stencilZFail=this.stencilZFail),this.stencilZPass!==Ns&&(r.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(r.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(r.rotation=this.rotation),this.polygonOffset===!0&&(r.polygonOffset=!0),this.polygonOffsetFactor!==0&&(r.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(r.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(r.linewidth=this.linewidth),this.dashSize!==void 0&&(r.dashSize=this.dashSize),this.gapSize!==void 0&&(r.gapSize=this.gapSize),this.scale!==void 0&&(r.scale=this.scale),this.dithering===!0&&(r.dithering=!0),this.alphaTest>0&&(r.alphaTest=this.alphaTest),this.alphaHash===!0&&(r.alphaHash=!0),this.alphaToCoverage===!0&&(r.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(r.premultipliedAlpha=!0),this.forceSinglePass===!0&&(r.forceSinglePass=!0),this.allowOverride===!1&&(r.allowOverride=!1),this.wireframe===!0&&(r.wireframe=!0),this.wireframeLinewidth>1&&(r.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(r.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(r.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(r.flatShading=!0),this.visible===!1&&(r.visible=!1),this.toneMapped===!1&&(r.toneMapped=!1),this.fog===!1&&(r.fog=!1),Object.keys(this.userData).length>0&&(r.userData=this.userData);function a(l){const d=[];for(const m in l){const g=l[m];delete g.metadata,d.push(g)}return d}if(t){const l=a(e.textures),d=a(e.images);l.length>0&&(r.textures=l),d.length>0&&(r.images=d)}return r}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let r=null;if(t!==null){const a=t.length;r=new Array(a);for(let l=0;l!==a;++l)r[l]=t[l].clone()}return this.clippingPlanes=r,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}const $i=new oe,lf=new oe,yl=new oe,Cr=new oe,uf=new oe,Ml=new oe,cf=new oe;class b_{constructor(e=new oe,t=new oe(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,$i)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const r=t.dot(this.direction);return r<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,r)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=$i.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):($i.copy(this.origin).addScaledVector(this.direction,t),$i.distanceToSquared(e))}distanceSqToSegment(e,t,r,a){lf.copy(e).add(t).multiplyScalar(.5),yl.copy(t).sub(e).normalize(),Cr.copy(this.origin).sub(lf);const l=e.distanceTo(t)*.5,d=-this.direction.dot(yl),m=Cr.dot(this.direction),g=-Cr.dot(yl),_=Cr.lengthSq(),M=Math.abs(1-d*d);let u,f,p,y;if(M>0)if(u=d*g-m,f=d*m-g,y=l*M,u>=0)if(f>=-y)if(f<=y){const E=1/M;u*=E,f*=E,p=u*(u+d*f+2*m)+f*(d*u+f+2*g)+_}else f=l,u=Math.max(0,-(d*f+m)),p=-u*u+f*(f+2*g)+_;else f=-l,u=Math.max(0,-(d*f+m)),p=-u*u+f*(f+2*g)+_;else f<=-y?(u=Math.max(0,-(-d*l+m)),f=u>0?-l:Math.min(Math.max(-l,-g),l),p=-u*u+f*(f+2*g)+_):f<=y?(u=0,f=Math.min(Math.max(-l,-g),l),p=f*(f+2*g)+_):(u=Math.max(0,-(d*l+m)),f=u>0?l:Math.min(Math.max(-l,-g),l),p=-u*u+f*(f+2*g)+_);else f=d>0?-l:l,u=Math.max(0,-(d*f+m)),p=-u*u+f*(f+2*g)+_;return r&&r.copy(this.origin).addScaledVector(this.direction,u),a&&a.copy(lf).addScaledVector(yl,f),p}intersectSphere(e,t){$i.subVectors(e.center,this.origin);const r=$i.dot(this.direction),a=$i.dot($i)-r*r,l=e.radius*e.radius;if(a>l)return null;const d=Math.sqrt(l-a),m=r-d,g=r+d;return g<0?null:m<0?this.at(g,t):this.at(m,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const r=-(this.origin.dot(e.normal)+e.constant)/t;return r>=0?r:null}intersectPlane(e,t){const r=this.distanceToPlane(e);return r===null?null:this.at(r,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let r,a,l,d,m,g;const _=1/this.direction.x,M=1/this.direction.y,u=1/this.direction.z,f=this.origin;return _>=0?(r=(e.min.x-f.x)*_,a=(e.max.x-f.x)*_):(r=(e.max.x-f.x)*_,a=(e.min.x-f.x)*_),M>=0?(l=(e.min.y-f.y)*M,d=(e.max.y-f.y)*M):(l=(e.max.y-f.y)*M,d=(e.min.y-f.y)*M),r>d||l>a||((l>r||isNaN(r))&&(r=l),(d<a||isNaN(a))&&(a=d),u>=0?(m=(e.min.z-f.z)*u,g=(e.max.z-f.z)*u):(m=(e.max.z-f.z)*u,g=(e.min.z-f.z)*u),r>g||m>a)||((m>r||r!==r)&&(r=m),(g<a||a!==a)&&(a=g),a<0)?null:this.at(r>=0?r:a,t)}intersectsBox(e){return this.intersectBox(e,$i)!==null}intersectTriangle(e,t,r,a,l){uf.subVectors(t,e),Ml.subVectors(r,e),cf.crossVectors(uf,Ml);let d=this.direction.dot(cf),m;if(d>0){if(a)return null;m=1}else if(d<0)m=-1,d=-d;else return null;Cr.subVectors(this.origin,e);const g=m*this.direction.dot(Ml.crossVectors(Cr,Ml));if(g<0)return null;const _=m*this.direction.dot(uf.cross(Cr));if(_<0||g+_>d)return null;const M=-m*Cr.dot(cf);return M<0?null:this.at(M/d,l)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class P_ extends aa{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new At(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new as,this.combine=a_,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Tm=new rn,$r=new b_,El=new eu,wm=new oe,Tl=new oe,wl=new oe,Al=new oe,ff=new oe,Rl=new oe,Am=new oe,Cl=new oe;class rr extends zn{constructor(e=new ri,t=new P_){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,r=Object.keys(t);if(r.length>0){const a=t[r[0]];if(a!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let l=0,d=a.length;l<d;l++){const m=a[l].name||String(l);this.morphTargetInfluences.push(0),this.morphTargetDictionary[m]=l}}}}getVertexPosition(e,t){const r=this.geometry,a=r.attributes.position,l=r.morphAttributes.position,d=r.morphTargetsRelative;t.fromBufferAttribute(a,e);const m=this.morphTargetInfluences;if(l&&m){Rl.set(0,0,0);for(let g=0,_=l.length;g<_;g++){const M=m[g],u=l[g];M!==0&&(ff.fromBufferAttribute(u,e),d?Rl.addScaledVector(ff,M):Rl.addScaledVector(ff.sub(t),M))}t.add(Rl)}return t}raycast(e,t){const r=this.geometry,a=this.material,l=this.matrixWorld;a!==void 0&&(r.boundingSphere===null&&r.computeBoundingSphere(),El.copy(r.boundingSphere),El.applyMatrix4(l),$r.copy(e.ray).recast(e.near),!(El.containsPoint($r.origin)===!1&&($r.intersectSphere(El,wm)===null||$r.origin.distanceToSquared(wm)>(e.far-e.near)**2))&&(Tm.copy(l).invert(),$r.copy(e.ray).applyMatrix4(Tm),!(r.boundingBox!==null&&$r.intersectsBox(r.boundingBox)===!1)&&this._computeIntersections(e,t,$r)))}_computeIntersections(e,t,r){let a;const l=this.geometry,d=this.material,m=l.index,g=l.attributes.position,_=l.attributes.uv,M=l.attributes.uv1,u=l.attributes.normal,f=l.groups,p=l.drawRange;if(m!==null)if(Array.isArray(d))for(let y=0,E=f.length;y<E;y++){const S=f[y],v=d[S.materialIndex],A=Math.max(S.start,p.start),P=Math.min(m.count,Math.min(S.start+S.count,p.start+p.count));for(let L=A,z=P;L<z;L+=3){const D=m.getX(L),F=m.getX(L+1),R=m.getX(L+2);a=bl(this,v,e,r,_,M,u,D,F,R),a&&(a.faceIndex=Math.floor(L/3),a.face.materialIndex=S.materialIndex,t.push(a))}}else{const y=Math.max(0,p.start),E=Math.min(m.count,p.start+p.count);for(let S=y,v=E;S<v;S+=3){const A=m.getX(S),P=m.getX(S+1),L=m.getX(S+2);a=bl(this,d,e,r,_,M,u,A,P,L),a&&(a.faceIndex=Math.floor(S/3),t.push(a))}}else if(g!==void 0)if(Array.isArray(d))for(let y=0,E=f.length;y<E;y++){const S=f[y],v=d[S.materialIndex],A=Math.max(S.start,p.start),P=Math.min(g.count,Math.min(S.start+S.count,p.start+p.count));for(let L=A,z=P;L<z;L+=3){const D=L,F=L+1,R=L+2;a=bl(this,v,e,r,_,M,u,D,F,R),a&&(a.faceIndex=Math.floor(L/3),a.face.materialIndex=S.materialIndex,t.push(a))}}else{const y=Math.max(0,p.start),E=Math.min(g.count,p.start+p.count);for(let S=y,v=E;S<v;S+=3){const A=S,P=S+1,L=S+2;a=bl(this,d,e,r,_,M,u,A,P,L),a&&(a.faceIndex=Math.floor(S/3),t.push(a))}}}}function $v(s,e,t,r,a,l,d,m){let g;if(e.side===kn?g=r.intersectTriangle(d,l,a,!0,m):g=r.intersectTriangle(a,l,d,e.side===Dr,m),g===null)return null;Cl.copy(m),Cl.applyMatrix4(s.matrixWorld);const _=t.ray.origin.distanceTo(Cl);return _<t.near||_>t.far?null:{distance:_,point:Cl.clone(),object:s}}function bl(s,e,t,r,a,l,d,m,g,_){s.getVertexPosition(m,Tl),s.getVertexPosition(g,wl),s.getVertexPosition(_,Al);const M=$v(s,e,t,r,Tl,wl,Al,Am);if(M){const u=new oe;_i.getBarycoord(Am,Tl,wl,Al,u),a&&(M.uv=_i.getInterpolatedAttribute(a,m,g,_,u,new It)),l&&(M.uv1=_i.getInterpolatedAttribute(l,m,g,_,u,new It)),d&&(M.normal=_i.getInterpolatedAttribute(d,m,g,_,u,new oe),M.normal.dot(r.direction)>0&&M.normal.multiplyScalar(-1));const f={a:m,b:g,c:_,normal:new oe,materialIndex:0};_i.getNormal(Tl,wl,Al,f.normal),M.face=f,M.barycoord=u}return M}class Zv extends Pn{constructor(e=null,t=1,r=1,a,l,d,m,g,_=gn,M=gn,u,f){super(null,d,m,g,_,M,a,l,u,f),this.isDataTexture=!0,this.image={data:e,width:t,height:r},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const df=new oe,Qv=new oe,Jv=new lt;class Jr{constructor(e=new oe(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,r,a){return this.normal.set(e,t,r),this.constant=a,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,r){const a=df.subVectors(r,t).cross(Qv.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(a,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t,r=!0){const a=e.delta(df),l=this.normal.dot(a);if(l===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const d=-(e.start.dot(this.normal)+this.constant)/l;return r===!0&&(d<0||d>1)?null:t.copy(e.start).addScaledVector(a,d)}intersectsLine(e){const t=this.distanceToPoint(e.start),r=this.distanceToPoint(e.end);return t<0&&r>0||r<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const r=t||Jv.getNormalMatrix(e),a=this.coplanarPoint(df).applyMatrix4(e),l=this.normal.applyMatrix3(r).normalize();return this.constant=-a.dot(l),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Zr=new eu,ex=new It(.5,.5),Pl=new oe;class L_{constructor(e=new Jr,t=new Jr,r=new Jr,a=new Jr,l=new Jr,d=new Jr){this.planes=[e,t,r,a,l,d]}set(e,t,r,a,l,d){const m=this.planes;return m[0].copy(e),m[1].copy(t),m[2].copy(r),m[3].copy(a),m[4].copy(l),m[5].copy(d),this}copy(e){const t=this.planes;for(let r=0;r<6;r++)t[r].copy(e.planes[r]);return this}setFromProjectionMatrix(e,t=Li,r=!1){const a=this.planes,l=e.elements,d=l[0],m=l[1],g=l[2],_=l[3],M=l[4],u=l[5],f=l[6],p=l[7],y=l[8],E=l[9],S=l[10],v=l[11],A=l[12],P=l[13],L=l[14],z=l[15];if(a[0].setComponents(_-d,p-M,v-y,z-A).normalize(),a[1].setComponents(_+d,p+M,v+y,z+A).normalize(),a[2].setComponents(_+m,p+u,v+E,z+P).normalize(),a[3].setComponents(_-m,p-u,v-E,z-P).normalize(),r)a[4].setComponents(g,f,S,L).normalize(),a[5].setComponents(_-g,p-f,v-S,z-L).normalize();else if(a[4].setComponents(_-g,p-f,v-S,z-L).normalize(),t===Li)a[5].setComponents(_+g,p+f,v+S,z+L).normalize();else if(t===$l)a[5].setComponents(g,f,S,L).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Zr.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),Zr.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Zr)}intersectsSprite(e){Zr.center.set(0,0,0);const t=ex.distanceTo(e.center);return Zr.radius=.7071067811865476+t,Zr.applyMatrix4(e.matrixWorld),this.intersectsSphere(Zr)}intersectsSphere(e){const t=this.planes,r=e.center,a=-e.radius;for(let l=0;l<6;l++)if(t[l].distanceToPoint(r)<a)return!1;return!0}intersectsBox(e){const t=this.planes;for(let r=0;r<6;r++){const a=t[r];if(Pl.x=a.normal.x>0?e.max.x:e.min.x,Pl.y=a.normal.y>0?e.max.y:e.min.y,Pl.z=a.normal.z>0?e.max.z:e.min.z,a.distanceToPoint(Pl)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let r=0;r<6;r++)if(t[r].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class D_ extends aa{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new At(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const Rm=new rn,cd=new b_,Ll=new eu,Dl=new oe;class Cm extends zn{constructor(e=new ri,t=new D_){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const r=this.geometry,a=this.matrixWorld,l=e.params.Points.threshold,d=r.drawRange;if(r.boundingSphere===null&&r.computeBoundingSphere(),Ll.copy(r.boundingSphere),Ll.applyMatrix4(a),Ll.radius+=l,e.ray.intersectsSphere(Ll)===!1)return;Rm.copy(a).invert(),cd.copy(e.ray).applyMatrix4(Rm);const m=l/((this.scale.x+this.scale.y+this.scale.z)/3),g=m*m,_=r.index,u=r.attributes.position;if(_!==null){const f=Math.max(0,d.start),p=Math.min(_.count,d.start+d.count);for(let y=f,E=p;y<E;y++){const S=_.getX(y);Dl.fromBufferAttribute(u,S),bm(Dl,S,g,a,e,t,this)}}else{const f=Math.max(0,d.start),p=Math.min(u.count,d.start+d.count);for(let y=f,E=p;y<E;y++)Dl.fromBufferAttribute(u,y),bm(Dl,y,g,a,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,r=Object.keys(t);if(r.length>0){const a=t[r[0]];if(a!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let l=0,d=a.length;l<d;l++){const m=a[l].name||String(l);this.morphTargetInfluences.push(0),this.morphTargetDictionary[m]=l}}}}}function bm(s,e,t,r,a,l,d){const m=cd.distanceSqToPoint(s);if(m<t){const g=new oe;cd.closestPointToPoint(s,g),g.applyMatrix4(r);const _=a.ray.origin.distanceTo(g);if(_<a.near||_>a.far)return;l.push({distance:_,distanceToRay:Math.sqrt(m),point:g,index:e,face:null,faceIndex:null,barycoord:null,object:d})}}class I_ extends Pn{constructor(e=[],t=ss,r,a,l,d,m,g,_,M){super(e,t,r,a,l,d,m,g,_,M),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class Js extends Pn{constructor(e,t,r=Ni,a,l,d,m=gn,g=gn,_,M=ir,u=1){if(M!==ir&&M!==is)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const f={width:e,height:t,depth:u};super(f,a,l,d,m,g,M,r,_),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Td(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class tx extends Js{constructor(e,t=Ni,r=ss,a,l,d=gn,m=gn,g,_=ir){const M={width:e,height:e,depth:1},u=[M,M,M,M,M,M];super(e,e,t,r,a,l,d,m,g,_),this.image=u,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}}class N_ extends Pn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class la extends ri{constructor(e=1,t=1,r=1,a=1,l=1,d=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:r,widthSegments:a,heightSegments:l,depthSegments:d};const m=this;a=Math.floor(a),l=Math.floor(l),d=Math.floor(d);const g=[],_=[],M=[],u=[];let f=0,p=0;y("z","y","x",-1,-1,r,t,e,d,l,0),y("z","y","x",1,-1,r,t,-e,d,l,1),y("x","z","y",1,1,e,r,t,a,d,2),y("x","z","y",1,-1,e,r,-t,a,d,3),y("x","y","z",1,-1,e,t,r,a,l,4),y("x","y","z",-1,-1,e,t,-r,a,l,5),this.setIndex(g),this.setAttribute("position",new tr(_,3)),this.setAttribute("normal",new tr(M,3)),this.setAttribute("uv",new tr(u,2));function y(E,S,v,A,P,L,z,D,F,R,I){const W=L/F,O=z/R,j=L/2,re=z/2,ae=D/2,X=F+1,Z=R+1;let q=0,G=0;const J=new oe;for(let ie=0;ie<Z;ie++){const U=ie*O-re;for(let K=0;K<X;K++){const Le=K*W-j;J[E]=Le*A,J[S]=U*P,J[v]=ae,_.push(J.x,J.y,J.z),J[E]=0,J[S]=0,J[v]=D>0?1:-1,M.push(J.x,J.y,J.z),u.push(K/F),u.push(1-ie/R),q+=1}}for(let ie=0;ie<R;ie++)for(let U=0;U<F;U++){const K=f+U+X*ie,Le=f+U+X*(ie+1),De=f+(U+1)+X*(ie+1),we=f+(U+1)+X*ie;g.push(K,Le,we),g.push(Le,De,we),G+=6}m.addGroup(p,G,I),p+=G,f+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new la(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}class tu extends ri{constructor(e=1,t=1,r=1,a=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:r,heightSegments:a};const l=e/2,d=t/2,m=Math.floor(r),g=Math.floor(a),_=m+1,M=g+1,u=e/m,f=t/g,p=[],y=[],E=[],S=[];for(let v=0;v<M;v++){const A=v*f-d;for(let P=0;P<_;P++){const L=P*u-l;y.push(L,-A,0),E.push(0,0,1),S.push(P/m),S.push(1-v/g)}}for(let v=0;v<g;v++)for(let A=0;A<m;A++){const P=A+_*v,L=A+_*(v+1),z=A+1+_*(v+1),D=A+1+_*v;p.push(P,L,D),p.push(L,z,D)}this.setIndex(p),this.setAttribute("position",new tr(y,3)),this.setAttribute("normal",new tr(E,3)),this.setAttribute("uv",new tr(S,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new tu(e.width,e.height,e.widthSegments,e.heightSegments)}}function eo(s){const e={};for(const t in s){e[t]={};for(const r in s[t]){const a=s[t][r];if(Pm(a))a.isRenderTargetTexture?(tt("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][r]=null):e[t][r]=a.clone();else if(Array.isArray(a))if(Pm(a[0])){const l=[];for(let d=0,m=a.length;d<m;d++)l[d]=a[d].clone();e[t][r]=l}else e[t][r]=a.slice();else e[t][r]=a}}return e}function bn(s){const e={};for(let t=0;t<s.length;t++){const r=eo(s[t]);for(const a in r)e[a]=r[a]}return e}function Pm(s){return s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)}function nx(s){const e=[];for(let t=0;t<s.length;t++)e.push(s[t].clone());return e}function U_(s){const e=s.getRenderTarget();return e===null?s.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:xt.workingColorSpace}const ix={clone:eo,merge:bn};var rx=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,sx=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class xi extends aa{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=rx,this.fragmentShader=sx,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=eo(e.uniforms),this.uniformsGroups=nx(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const a in this.uniforms){const d=this.uniforms[a].value;d&&d.isTexture?t.uniforms[a]={type:"t",value:d.toJSON(e).uuid}:d&&d.isColor?t.uniforms[a]={type:"c",value:d.getHex()}:d&&d.isVector2?t.uniforms[a]={type:"v2",value:d.toArray()}:d&&d.isVector3?t.uniforms[a]={type:"v3",value:d.toArray()}:d&&d.isVector4?t.uniforms[a]={type:"v4",value:d.toArray()}:d&&d.isMatrix3?t.uniforms[a]={type:"m3",value:d.toArray()}:d&&d.isMatrix4?t.uniforms[a]={type:"m4",value:d.toArray()}:t.uniforms[a]={value:d}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const r={};for(const a in this.extensions)this.extensions[a]===!0&&(r[a]=!0);return Object.keys(r).length>0&&(t.extensions=r),t}}class ox extends xi{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}}class ax extends aa{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=iv,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class lx extends aa{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}const Il=new oe,Nl=new io,Ai=new oe;class F_ extends zn{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new rn,this.projectionMatrix=new rn,this.projectionMatrixInverse=new rn,this.coordinateSystem=Li,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(Il,Nl,Ai),Ai.x===1&&Ai.y===1&&Ai.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Il,Nl,Ai.set(1,1,1)).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorld.decompose(Il,Nl,Ai),Ai.x===1&&Ai.y===1&&Ai.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Il,Nl,Ai.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}}const br=new oe,Lm=new It,Dm=new It;class ni extends F_{constructor(e=50,t=1,r=.1,a=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=r,this.far=a,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=ra*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(ea*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return ra*2*Math.atan(Math.tan(ea*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,r){br.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(br.x,br.y).multiplyScalar(-e/br.z),br.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),r.set(br.x,br.y).multiplyScalar(-e/br.z)}getViewSize(e,t){return this.getViewBounds(e,Lm,Dm),t.subVectors(Dm,Lm)}setViewOffset(e,t,r,a,l,d){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=r,this.view.offsetY=a,this.view.width=l,this.view.height=d,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(ea*.5*this.fov)/this.zoom,r=2*t,a=this.aspect*r,l=-.5*a;const d=this.view;if(this.view!==null&&this.view.enabled){const g=d.fullWidth,_=d.fullHeight;l+=d.offsetX*a/g,t-=d.offsetY*r/_,a*=d.width/g,r*=d.height/_}const m=this.filmOffset;m!==0&&(l+=e*m/this.getFilmWidth()),this.projectionMatrix.makePerspective(l,l+a,t,t-r,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}class O_ extends F_{constructor(e=-1,t=1,r=1,a=-1,l=.1,d=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=r,this.bottom=a,this.near=l,this.far=d,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,r,a,l,d){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=r,this.view.offsetY=a,this.view.width=l,this.view.height=d,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),r=(this.right+this.left)/2,a=(this.top+this.bottom)/2;let l=r-e,d=r+e,m=a+t,g=a-t;if(this.view!==null&&this.view.enabled){const _=(this.right-this.left)/this.view.fullWidth/this.zoom,M=(this.top-this.bottom)/this.view.fullHeight/this.zoom;l+=_*this.view.offsetX,d=l+_*this.view.width,m-=M*this.view.offsetY,g=m-M*this.view.height}this.projectionMatrix.makeOrthographic(l,d,m,g,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}const Xs=-90,Ys=1;class ux extends zn{constructor(e,t,r){super(),this.type="CubeCamera",this.renderTarget=r,this.coordinateSystem=null,this.activeMipmapLevel=0;const a=new ni(Xs,Ys,e,t);a.layers=this.layers,this.add(a);const l=new ni(Xs,Ys,e,t);l.layers=this.layers,this.add(l);const d=new ni(Xs,Ys,e,t);d.layers=this.layers,this.add(d);const m=new ni(Xs,Ys,e,t);m.layers=this.layers,this.add(m);const g=new ni(Xs,Ys,e,t);g.layers=this.layers,this.add(g);const _=new ni(Xs,Ys,e,t);_.layers=this.layers,this.add(_)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[r,a,l,d,m,g]=t;for(const _ of t)this.remove(_);if(e===Li)r.up.set(0,1,0),r.lookAt(1,0,0),a.up.set(0,1,0),a.lookAt(-1,0,0),l.up.set(0,0,-1),l.lookAt(0,1,0),d.up.set(0,0,1),d.lookAt(0,-1,0),m.up.set(0,1,0),m.lookAt(0,0,1),g.up.set(0,1,0),g.lookAt(0,0,-1);else if(e===$l)r.up.set(0,-1,0),r.lookAt(-1,0,0),a.up.set(0,-1,0),a.lookAt(1,0,0),l.up.set(0,0,1),l.lookAt(0,1,0),d.up.set(0,0,-1),d.lookAt(0,-1,0),m.up.set(0,-1,0),m.lookAt(0,0,1),g.up.set(0,-1,0),g.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const _ of t)this.add(_),_.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:r,activeMipmapLevel:a}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[l,d,m,g,_,M]=this.children,u=e.getRenderTarget(),f=e.getActiveCubeFace(),p=e.getActiveMipmapLevel(),y=e.xr.enabled;e.xr.enabled=!1;const E=r.texture.generateMipmaps;r.texture.generateMipmaps=!1;let S=!1;e.isWebGLRenderer===!0?S=e.state.buffers.depth.getReversed():S=e.reversedDepthBuffer,e.setRenderTarget(r,0,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),e.setRenderTarget(r,1,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,d),e.setRenderTarget(r,2,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,m),e.setRenderTarget(r,3,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,g),e.setRenderTarget(r,4,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,_),r.texture.generateMipmaps=E,e.setRenderTarget(r,5,a),S&&e.autoClear===!1&&e.clearDepth(),e.render(t,M),e.setRenderTarget(u,f,p),e.xr.enabled=y,r.texture.needsPMREMUpdate=!0}}class cx extends ni{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}const Pd=class Pd{constructor(e,t,r,a){this.elements=[1,0,0,1],e!==void 0&&this.set(e,t,r,a)}identity(){return this.set(1,0,0,1),this}fromArray(e,t=0){for(let r=0;r<4;r++)this.elements[r]=e[r+t];return this}set(e,t,r,a){const l=this.elements;return l[0]=e,l[2]=t,l[1]=r,l[3]=a,this}};Pd.prototype.isMatrix2=!0;let Im=Pd;function Nm(s,e,t,r){const a=fx(r);switch(t){case S_:return s*e;case M_:return s*e/a.components*a.byteLength;case vd:return s*e/a.components*a.byteLength;case os:return s*e*2/a.components*a.byteLength;case xd:return s*e*2/a.components*a.byteLength;case y_:return s*e*3/a.components*a.byteLength;case gi:return s*e*4/a.components*a.byteLength;case Sd:return s*e*4/a.components*a.byteLength;case zl:case Hl:return Math.floor((s+3)/4)*Math.floor((e+3)/4)*8;case Vl:case Gl:return Math.floor((s+3)/4)*Math.floor((e+3)/4)*16;case Nf:case Ff:return Math.max(s,16)*Math.max(e,8)/4;case If:case Uf:return Math.max(s,8)*Math.max(e,8)/2;case Of:case Bf:case zf:case Hf:return Math.floor((s+3)/4)*Math.floor((e+3)/4)*8;case kf:case Yl:case Vf:return Math.floor((s+3)/4)*Math.floor((e+3)/4)*16;case Gf:return Math.floor((s+3)/4)*Math.floor((e+3)/4)*16;case Wf:return Math.floor((s+4)/5)*Math.floor((e+3)/4)*16;case Xf:return Math.floor((s+4)/5)*Math.floor((e+4)/5)*16;case Yf:return Math.floor((s+5)/6)*Math.floor((e+4)/5)*16;case qf:return Math.floor((s+5)/6)*Math.floor((e+5)/6)*16;case jf:return Math.floor((s+7)/8)*Math.floor((e+4)/5)*16;case Kf:return Math.floor((s+7)/8)*Math.floor((e+5)/6)*16;case $f:return Math.floor((s+7)/8)*Math.floor((e+7)/8)*16;case Zf:return Math.floor((s+9)/10)*Math.floor((e+4)/5)*16;case Qf:return Math.floor((s+9)/10)*Math.floor((e+5)/6)*16;case Jf:return Math.floor((s+9)/10)*Math.floor((e+7)/8)*16;case ed:return Math.floor((s+9)/10)*Math.floor((e+9)/10)*16;case td:return Math.floor((s+11)/12)*Math.floor((e+9)/10)*16;case nd:return Math.floor((s+11)/12)*Math.floor((e+11)/12)*16;case id:case rd:case sd:return Math.ceil(s/4)*Math.ceil(e/4)*16;case od:case ad:return Math.ceil(s/4)*Math.ceil(e/4)*8;case ql:case ld:return Math.ceil(s/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function fx(s){switch(s){case ii:case __:return{byteLength:1,components:1};case na:case g_:case nr:return{byteLength:2,components:1};case _d:case gd:return{byteLength:2,components:4};case Ni:case md:case Pi:return{byteLength:4,components:1};case v_:case x_:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${s}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:pd}}));typeof window<"u"&&(window.__THREE__?tt("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=pd);/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function B_(){let s=null,e=!1,t=null,r=null;function a(l,d){t(l,d),r=s.requestAnimationFrame(a)}return{start:function(){e!==!0&&t!==null&&s!==null&&(r=s.requestAnimationFrame(a),e=!0)},stop:function(){s!==null&&s.cancelAnimationFrame(r),e=!1},setAnimationLoop:function(l){t=l},setContext:function(l){s=l}}}function dx(s){const e=new WeakMap;function t(m,g){const _=m.array,M=m.usage,u=_.byteLength,f=s.createBuffer();s.bindBuffer(g,f),s.bufferData(g,_,M),m.onUploadCallback();let p;if(_ instanceof Float32Array)p=s.FLOAT;else if(typeof Float16Array<"u"&&_ instanceof Float16Array)p=s.HALF_FLOAT;else if(_ instanceof Uint16Array)m.isFloat16BufferAttribute?p=s.HALF_FLOAT:p=s.UNSIGNED_SHORT;else if(_ instanceof Int16Array)p=s.SHORT;else if(_ instanceof Uint32Array)p=s.UNSIGNED_INT;else if(_ instanceof Int32Array)p=s.INT;else if(_ instanceof Int8Array)p=s.BYTE;else if(_ instanceof Uint8Array)p=s.UNSIGNED_BYTE;else if(_ instanceof Uint8ClampedArray)p=s.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+_);return{buffer:f,type:p,bytesPerElement:_.BYTES_PER_ELEMENT,version:m.version,size:u}}function r(m,g,_){const M=g.array,u=g.updateRanges;if(s.bindBuffer(_,m),u.length===0)s.bufferSubData(_,0,M);else{u.sort((p,y)=>p.start-y.start);let f=0;for(let p=1;p<u.length;p++){const y=u[f],E=u[p];E.start<=y.start+y.count+1?y.count=Math.max(y.count,E.start+E.count-y.start):(++f,u[f]=E)}u.length=f+1;for(let p=0,y=u.length;p<y;p++){const E=u[p];s.bufferSubData(_,E.start*M.BYTES_PER_ELEMENT,M,E.start,E.count)}g.clearUpdateRanges()}g.onUploadCallback()}function a(m){return m.isInterleavedBufferAttribute&&(m=m.data),e.get(m)}function l(m){m.isInterleavedBufferAttribute&&(m=m.data);const g=e.get(m);g&&(s.deleteBuffer(g.buffer),e.delete(m))}function d(m,g){if(m.isInterleavedBufferAttribute&&(m=m.data),m.isGLBufferAttribute){const M=e.get(m);(!M||M.version<m.version)&&e.set(m,{buffer:m.buffer,type:m.type,bytesPerElement:m.elementSize,version:m.version});return}const _=e.get(m);if(_===void 0)e.set(m,t(m,g));else if(_.version<m.version){if(_.size!==m.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");r(_.buffer,m,g),_.version=m.version}}return{get:a,remove:l,update:d}}var hx=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,px=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,mx=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,_x=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,gx=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,vx=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,xx=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Sx=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,yx=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,Mx=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Ex=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Tx=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,wx=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Ax=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Rx=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Cx=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,bx=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Px=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Lx=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Dx=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,Ix=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,Nx=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,Ux=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,Fx=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Ox=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Bx=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,kx=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,zx=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Hx=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Vx=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Gx="gl_FragColor = linearToOutputTexel( gl_FragColor );",Wx=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Xx=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,Yx=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,qx=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,jx=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,Kx=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,$x=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,Zx=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,Qx=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Jx=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,eS=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,tS=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,nS=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,iS=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,rS=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,sS=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,oS=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,aS=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,lS=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,uS=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,cS=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,fS=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,dS=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#ifdef USE_LIGHT_PROBES_GRID
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = inverseTransformDirection( geometryNormal, viewMatrix );
		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,hS=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,pS=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,mS=`#ifdef USE_LIGHT_PROBES_GRID
uniform highp sampler3D probesSH;
uniform vec3 probesMin;
uniform vec3 probesMax;
uniform vec3 probesResolution;
vec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 resMinusOne = res - 1.0;
	vec3 probeSpacing = gridRange / resMinusOne;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );
	uvw = uvw * resMinusOne / res + 0.5 / res;
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,_S=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,gS=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,vS=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,xS=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,SS=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,yS=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,MS=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,ES=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,TS=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,wS=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,AS=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,RS=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,CS=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,bS=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,PS=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,LS=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,DS=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,IS=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,NS=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,US=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,FS=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,OS=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,BS=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,kS=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,zS=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,HS=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,VS=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,GS=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,WS=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,XS=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,YS=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,qS=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,jS=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,KS=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,$S=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,ZS=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,QS=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,JS=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,ey=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,ty=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,ny=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,iy=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,ry=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,sy=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,oy=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,ay=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,ly=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,uy=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,cy=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,fy=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,dy=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const hy=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,py=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,my=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,_y=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,gy=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,vy=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,xy=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Sy=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,yy=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,My=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,Ey=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Ty=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,wy=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Ay=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Ry=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Cy=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,by=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Py=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Ly=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Dy=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Iy=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Ny=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Uy=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Fy=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Oy=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,By=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ky=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,zy=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Hy=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Vy=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Gy=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Wy=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Xy=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Yy=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ht={alphahash_fragment:hx,alphahash_pars_fragment:px,alphamap_fragment:mx,alphamap_pars_fragment:_x,alphatest_fragment:gx,alphatest_pars_fragment:vx,aomap_fragment:xx,aomap_pars_fragment:Sx,batching_pars_vertex:yx,batching_vertex:Mx,begin_vertex:Ex,beginnormal_vertex:Tx,bsdfs:wx,iridescence_fragment:Ax,bumpmap_pars_fragment:Rx,clipping_planes_fragment:Cx,clipping_planes_pars_fragment:bx,clipping_planes_pars_vertex:Px,clipping_planes_vertex:Lx,color_fragment:Dx,color_pars_fragment:Ix,color_pars_vertex:Nx,color_vertex:Ux,common:Fx,cube_uv_reflection_fragment:Ox,defaultnormal_vertex:Bx,displacementmap_pars_vertex:kx,displacementmap_vertex:zx,emissivemap_fragment:Hx,emissivemap_pars_fragment:Vx,colorspace_fragment:Gx,colorspace_pars_fragment:Wx,envmap_fragment:Xx,envmap_common_pars_fragment:Yx,envmap_pars_fragment:qx,envmap_pars_vertex:jx,envmap_physical_pars_fragment:sS,envmap_vertex:Kx,fog_vertex:$x,fog_pars_vertex:Zx,fog_fragment:Qx,fog_pars_fragment:Jx,gradientmap_pars_fragment:eS,lightmap_pars_fragment:tS,lights_lambert_fragment:nS,lights_lambert_pars_fragment:iS,lights_pars_begin:rS,lights_toon_fragment:oS,lights_toon_pars_fragment:aS,lights_phong_fragment:lS,lights_phong_pars_fragment:uS,lights_physical_fragment:cS,lights_physical_pars_fragment:fS,lights_fragment_begin:dS,lights_fragment_maps:hS,lights_fragment_end:pS,lightprobes_pars_fragment:mS,logdepthbuf_fragment:_S,logdepthbuf_pars_fragment:gS,logdepthbuf_pars_vertex:vS,logdepthbuf_vertex:xS,map_fragment:SS,map_pars_fragment:yS,map_particle_fragment:MS,map_particle_pars_fragment:ES,metalnessmap_fragment:TS,metalnessmap_pars_fragment:wS,morphinstance_vertex:AS,morphcolor_vertex:RS,morphnormal_vertex:CS,morphtarget_pars_vertex:bS,morphtarget_vertex:PS,normal_fragment_begin:LS,normal_fragment_maps:DS,normal_pars_fragment:IS,normal_pars_vertex:NS,normal_vertex:US,normalmap_pars_fragment:FS,clearcoat_normal_fragment_begin:OS,clearcoat_normal_fragment_maps:BS,clearcoat_pars_fragment:kS,iridescence_pars_fragment:zS,opaque_fragment:HS,packing:VS,premultiplied_alpha_fragment:GS,project_vertex:WS,dithering_fragment:XS,dithering_pars_fragment:YS,roughnessmap_fragment:qS,roughnessmap_pars_fragment:jS,shadowmap_pars_fragment:KS,shadowmap_pars_vertex:$S,shadowmap_vertex:ZS,shadowmask_pars_fragment:QS,skinbase_vertex:JS,skinning_pars_vertex:ey,skinning_vertex:ty,skinnormal_vertex:ny,specularmap_fragment:iy,specularmap_pars_fragment:ry,tonemapping_fragment:sy,tonemapping_pars_fragment:oy,transmission_fragment:ay,transmission_pars_fragment:ly,uv_pars_fragment:uy,uv_pars_vertex:cy,uv_vertex:fy,worldpos_vertex:dy,background_vert:hy,background_frag:py,backgroundCube_vert:my,backgroundCube_frag:_y,cube_vert:gy,cube_frag:vy,depth_vert:xy,depth_frag:Sy,distance_vert:yy,distance_frag:My,equirect_vert:Ey,equirect_frag:Ty,linedashed_vert:wy,linedashed_frag:Ay,meshbasic_vert:Ry,meshbasic_frag:Cy,meshlambert_vert:by,meshlambert_frag:Py,meshmatcap_vert:Ly,meshmatcap_frag:Dy,meshnormal_vert:Iy,meshnormal_frag:Ny,meshphong_vert:Uy,meshphong_frag:Fy,meshphysical_vert:Oy,meshphysical_frag:By,meshtoon_vert:ky,meshtoon_frag:zy,points_vert:Hy,points_frag:Vy,shadow_vert:Gy,shadow_frag:Wy,sprite_vert:Xy,sprite_frag:Yy},Ue={common:{diffuse:{value:new At(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new lt},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new lt}},envmap:{envMap:{value:null},envMapRotation:{value:new lt},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new lt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new lt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new lt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new lt},normalScale:{value:new It(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new lt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new lt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new lt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new lt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new At(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new oe},probesMax:{value:new oe},probesResolution:{value:new oe}},points:{diffuse:{value:new At(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0},uvTransform:{value:new lt}},sprite:{diffuse:{value:new At(16777215)},opacity:{value:1},center:{value:new It(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new lt},alphaMap:{value:null},alphaMapTransform:{value:new lt},alphaTest:{value:0}}},bi={basic:{uniforms:bn([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.fog]),vertexShader:ht.meshbasic_vert,fragmentShader:ht.meshbasic_frag},lambert:{uniforms:bn([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,Ue.lights,{emissive:{value:new At(0)},envMapIntensity:{value:1}}]),vertexShader:ht.meshlambert_vert,fragmentShader:ht.meshlambert_frag},phong:{uniforms:bn([Ue.common,Ue.specularmap,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,Ue.lights,{emissive:{value:new At(0)},specular:{value:new At(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:ht.meshphong_vert,fragmentShader:ht.meshphong_frag},standard:{uniforms:bn([Ue.common,Ue.envmap,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.roughnessmap,Ue.metalnessmap,Ue.fog,Ue.lights,{emissive:{value:new At(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ht.meshphysical_vert,fragmentShader:ht.meshphysical_frag},toon:{uniforms:bn([Ue.common,Ue.aomap,Ue.lightmap,Ue.emissivemap,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.gradientmap,Ue.fog,Ue.lights,{emissive:{value:new At(0)}}]),vertexShader:ht.meshtoon_vert,fragmentShader:ht.meshtoon_frag},matcap:{uniforms:bn([Ue.common,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,Ue.fog,{matcap:{value:null}}]),vertexShader:ht.meshmatcap_vert,fragmentShader:ht.meshmatcap_frag},points:{uniforms:bn([Ue.points,Ue.fog]),vertexShader:ht.points_vert,fragmentShader:ht.points_frag},dashed:{uniforms:bn([Ue.common,Ue.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ht.linedashed_vert,fragmentShader:ht.linedashed_frag},depth:{uniforms:bn([Ue.common,Ue.displacementmap]),vertexShader:ht.depth_vert,fragmentShader:ht.depth_frag},normal:{uniforms:bn([Ue.common,Ue.bumpmap,Ue.normalmap,Ue.displacementmap,{opacity:{value:1}}]),vertexShader:ht.meshnormal_vert,fragmentShader:ht.meshnormal_frag},sprite:{uniforms:bn([Ue.sprite,Ue.fog]),vertexShader:ht.sprite_vert,fragmentShader:ht.sprite_frag},background:{uniforms:{uvTransform:{value:new lt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ht.background_vert,fragmentShader:ht.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new lt}},vertexShader:ht.backgroundCube_vert,fragmentShader:ht.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ht.cube_vert,fragmentShader:ht.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ht.equirect_vert,fragmentShader:ht.equirect_frag},distance:{uniforms:bn([Ue.common,Ue.displacementmap,{referencePosition:{value:new oe},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ht.distance_vert,fragmentShader:ht.distance_frag},shadow:{uniforms:bn([Ue.lights,Ue.fog,{color:{value:new At(0)},opacity:{value:1}}]),vertexShader:ht.shadow_vert,fragmentShader:ht.shadow_frag}};bi.physical={uniforms:bn([bi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new lt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new lt},clearcoatNormalScale:{value:new It(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new lt},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new lt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new lt},sheen:{value:0},sheenColor:{value:new At(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new lt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new lt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new lt},transmissionSamplerSize:{value:new It},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new lt},attenuationDistance:{value:0},attenuationColor:{value:new At(0)},specularColor:{value:new At(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new lt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new lt},anisotropyVector:{value:new It},anisotropyMap:{value:null},anisotropyMapTransform:{value:new lt}}]),vertexShader:ht.meshphysical_vert,fragmentShader:ht.meshphysical_frag};const Ul={r:0,b:0,g:0},qy=new rn,k_=new lt;k_.set(-1,0,0,0,1,0,0,0,1);function jy(s,e,t,r,a,l){const d=new At(0);let m=a===!0?0:1,g,_,M=null,u=0,f=null;function p(A){let P=A.isScene===!0?A.background:null;if(P&&P.isTexture){const L=A.backgroundBlurriness>0;P=e.get(P,L)}return P}function y(A){let P=!1;const L=p(A);L===null?S(d,m):L&&L.isColor&&(S(L,1),P=!0);const z=s.xr.getEnvironmentBlendMode();z==="additive"?t.buffers.color.setClear(0,0,0,1,l):z==="alpha-blend"&&t.buffers.color.setClear(0,0,0,0,l),(s.autoClear||P)&&(t.buffers.depth.setTest(!0),t.buffers.depth.setMask(!0),t.buffers.color.setMask(!0),s.clear(s.autoClearColor,s.autoClearDepth,s.autoClearStencil))}function E(A,P){const L=p(P);L&&(L.isCubeTexture||L.mapping===Jl)?(_===void 0&&(_=new rr(new la(1,1,1),new xi({name:"BackgroundCubeMaterial",uniforms:eo(bi.backgroundCube.uniforms),vertexShader:bi.backgroundCube.vertexShader,fragmentShader:bi.backgroundCube.fragmentShader,side:kn,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),_.geometry.deleteAttribute("normal"),_.geometry.deleteAttribute("uv"),_.onBeforeRender=function(z,D,F){this.matrixWorld.copyPosition(F.matrixWorld)},Object.defineProperty(_.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),r.update(_)),_.material.uniforms.envMap.value=L,_.material.uniforms.backgroundBlurriness.value=P.backgroundBlurriness,_.material.uniforms.backgroundIntensity.value=P.backgroundIntensity,_.material.uniforms.backgroundRotation.value.setFromMatrix4(qy.makeRotationFromEuler(P.backgroundRotation)).transpose(),L.isCubeTexture&&L.isRenderTargetTexture===!1&&_.material.uniforms.backgroundRotation.value.premultiply(k_),_.material.toneMapped=xt.getTransfer(L.colorSpace)!==Lt,(M!==L||u!==L.version||f!==s.toneMapping)&&(_.material.needsUpdate=!0,M=L,u=L.version,f=s.toneMapping),_.layers.enableAll(),A.unshift(_,_.geometry,_.material,0,0,null)):L&&L.isTexture&&(g===void 0&&(g=new rr(new tu(2,2),new xi({name:"BackgroundMaterial",uniforms:eo(bi.background.uniforms),vertexShader:bi.background.vertexShader,fragmentShader:bi.background.fragmentShader,side:Dr,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),g.geometry.deleteAttribute("normal"),Object.defineProperty(g.material,"map",{get:function(){return this.uniforms.t2D.value}}),r.update(g)),g.material.uniforms.t2D.value=L,g.material.uniforms.backgroundIntensity.value=P.backgroundIntensity,g.material.toneMapped=xt.getTransfer(L.colorSpace)!==Lt,L.matrixAutoUpdate===!0&&L.updateMatrix(),g.material.uniforms.uvTransform.value.copy(L.matrix),(M!==L||u!==L.version||f!==s.toneMapping)&&(g.material.needsUpdate=!0,M=L,u=L.version,f=s.toneMapping),g.layers.enableAll(),A.unshift(g,g.geometry,g.material,0,0,null))}function S(A,P){A.getRGB(Ul,U_(s)),t.buffers.color.setClear(Ul.r,Ul.g,Ul.b,P,l)}function v(){_!==void 0&&(_.geometry.dispose(),_.material.dispose(),_=void 0),g!==void 0&&(g.geometry.dispose(),g.material.dispose(),g=void 0)}return{getClearColor:function(){return d},setClearColor:function(A,P=1){d.set(A),m=P,S(d,m)},getClearAlpha:function(){return m},setClearAlpha:function(A){m=A,S(d,m)},render:y,addToRenderList:E,dispose:v}}function Ky(s,e){const t=s.getParameter(s.MAX_VERTEX_ATTRIBS),r={},a=f(null);let l=a,d=!1;function m(O,j,re,ae,X){let Z=!1;const q=u(O,ae,re,j);l!==q&&(l=q,_(l.object)),Z=p(O,ae,re,X),Z&&y(O,ae,re,X),X!==null&&e.update(X,s.ELEMENT_ARRAY_BUFFER),(Z||d)&&(d=!1,L(O,j,re,ae),X!==null&&s.bindBuffer(s.ELEMENT_ARRAY_BUFFER,e.get(X).buffer))}function g(){return s.createVertexArray()}function _(O){return s.bindVertexArray(O)}function M(O){return s.deleteVertexArray(O)}function u(O,j,re,ae){const X=ae.wireframe===!0;let Z=r[j.id];Z===void 0&&(Z={},r[j.id]=Z);const q=O.isInstancedMesh===!0?O.id:0;let G=Z[q];G===void 0&&(G={},Z[q]=G);let J=G[re.id];J===void 0&&(J={},G[re.id]=J);let ie=J[X];return ie===void 0&&(ie=f(g()),J[X]=ie),ie}function f(O){const j=[],re=[],ae=[];for(let X=0;X<t;X++)j[X]=0,re[X]=0,ae[X]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:j,enabledAttributes:re,attributeDivisors:ae,object:O,attributes:{},index:null}}function p(O,j,re,ae){const X=l.attributes,Z=j.attributes;let q=0;const G=re.getAttributes();for(const J in G)if(G[J].location>=0){const U=X[J];let K=Z[J];if(K===void 0&&(J==="instanceMatrix"&&O.instanceMatrix&&(K=O.instanceMatrix),J==="instanceColor"&&O.instanceColor&&(K=O.instanceColor)),U===void 0||U.attribute!==K||K&&U.data!==K.data)return!0;q++}return l.attributesNum!==q||l.index!==ae}function y(O,j,re,ae){const X={},Z=j.attributes;let q=0;const G=re.getAttributes();for(const J in G)if(G[J].location>=0){let U=Z[J];U===void 0&&(J==="instanceMatrix"&&O.instanceMatrix&&(U=O.instanceMatrix),J==="instanceColor"&&O.instanceColor&&(U=O.instanceColor));const K={};K.attribute=U,U&&U.data&&(K.data=U.data),X[J]=K,q++}l.attributes=X,l.attributesNum=q,l.index=ae}function E(){const O=l.newAttributes;for(let j=0,re=O.length;j<re;j++)O[j]=0}function S(O){v(O,0)}function v(O,j){const re=l.newAttributes,ae=l.enabledAttributes,X=l.attributeDivisors;re[O]=1,ae[O]===0&&(s.enableVertexAttribArray(O),ae[O]=1),X[O]!==j&&(s.vertexAttribDivisor(O,j),X[O]=j)}function A(){const O=l.newAttributes,j=l.enabledAttributes;for(let re=0,ae=j.length;re<ae;re++)j[re]!==O[re]&&(s.disableVertexAttribArray(re),j[re]=0)}function P(O,j,re,ae,X,Z,q){q===!0?s.vertexAttribIPointer(O,j,re,X,Z):s.vertexAttribPointer(O,j,re,ae,X,Z)}function L(O,j,re,ae){E();const X=ae.attributes,Z=re.getAttributes(),q=j.defaultAttributeValues;for(const G in Z){const J=Z[G];if(J.location>=0){let ie=X[G];if(ie===void 0&&(G==="instanceMatrix"&&O.instanceMatrix&&(ie=O.instanceMatrix),G==="instanceColor"&&O.instanceColor&&(ie=O.instanceColor)),ie!==void 0){const U=ie.normalized,K=ie.itemSize,Le=e.get(ie);if(Le===void 0)continue;const De=Le.buffer,we=Le.type,se=Le.bytesPerElement,_e=we===s.INT||we===s.UNSIGNED_INT||ie.gpuType===md;if(ie.isInterleavedBufferAttribute){const de=ie.data,Ie=de.stride,je=ie.offset;if(de.isInstancedInterleavedBuffer){for(let $e=0;$e<J.locationSize;$e++)v(J.location+$e,de.meshPerAttribute);O.isInstancedMesh!==!0&&ae._maxInstanceCount===void 0&&(ae._maxInstanceCount=de.meshPerAttribute*de.count)}else for(let $e=0;$e<J.locationSize;$e++)S(J.location+$e);s.bindBuffer(s.ARRAY_BUFFER,De);for(let $e=0;$e<J.locationSize;$e++)P(J.location+$e,K/J.locationSize,we,U,Ie*se,(je+K/J.locationSize*$e)*se,_e)}else{if(ie.isInstancedBufferAttribute){for(let de=0;de<J.locationSize;de++)v(J.location+de,ie.meshPerAttribute);O.isInstancedMesh!==!0&&ae._maxInstanceCount===void 0&&(ae._maxInstanceCount=ie.meshPerAttribute*ie.count)}else for(let de=0;de<J.locationSize;de++)S(J.location+de);s.bindBuffer(s.ARRAY_BUFFER,De);for(let de=0;de<J.locationSize;de++)P(J.location+de,K/J.locationSize,we,U,K*se,K/J.locationSize*de*se,_e)}}else if(q!==void 0){const U=q[G];if(U!==void 0)switch(U.length){case 2:s.vertexAttrib2fv(J.location,U);break;case 3:s.vertexAttrib3fv(J.location,U);break;case 4:s.vertexAttrib4fv(J.location,U);break;default:s.vertexAttrib1fv(J.location,U)}}}}A()}function z(){I();for(const O in r){const j=r[O];for(const re in j){const ae=j[re];for(const X in ae){const Z=ae[X];for(const q in Z)M(Z[q].object),delete Z[q];delete ae[X]}}delete r[O]}}function D(O){if(r[O.id]===void 0)return;const j=r[O.id];for(const re in j){const ae=j[re];for(const X in ae){const Z=ae[X];for(const q in Z)M(Z[q].object),delete Z[q];delete ae[X]}}delete r[O.id]}function F(O){for(const j in r){const re=r[j];for(const ae in re){const X=re[ae];if(X[O.id]===void 0)continue;const Z=X[O.id];for(const q in Z)M(Z[q].object),delete Z[q];delete X[O.id]}}}function R(O){for(const j in r){const re=r[j],ae=O.isInstancedMesh===!0?O.id:0,X=re[ae];if(X!==void 0){for(const Z in X){const q=X[Z];for(const G in q)M(q[G].object),delete q[G];delete X[Z]}delete re[ae],Object.keys(re).length===0&&delete r[j]}}}function I(){W(),d=!0,l!==a&&(l=a,_(l.object))}function W(){a.geometry=null,a.program=null,a.wireframe=!1}return{setup:m,reset:I,resetDefaultState:W,dispose:z,releaseStatesOfGeometry:D,releaseStatesOfObject:R,releaseStatesOfProgram:F,initAttributes:E,enableAttribute:S,disableUnusedAttributes:A}}function $y(s,e,t){let r;function a(g){r=g}function l(g,_){s.drawArrays(r,g,_),t.update(_,r,1)}function d(g,_,M){M!==0&&(s.drawArraysInstanced(r,g,_,M),t.update(_,r,M))}function m(g,_,M){if(M===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(r,g,0,_,0,M);let f=0;for(let p=0;p<M;p++)f+=_[p];t.update(f,r,1)}this.setMode=a,this.render=l,this.renderInstances=d,this.renderMultiDraw=m}function Zy(s,e,t,r){let a;function l(){if(a!==void 0)return a;if(e.has("EXT_texture_filter_anisotropic")===!0){const F=e.get("EXT_texture_filter_anisotropic");a=s.getParameter(F.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else a=0;return a}function d(F){return!(F!==gi&&r.convert(F)!==s.getParameter(s.IMPLEMENTATION_COLOR_READ_FORMAT))}function m(F){const R=F===nr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(F!==ii&&r.convert(F)!==s.getParameter(s.IMPLEMENTATION_COLOR_READ_TYPE)&&F!==Pi&&!R)}function g(F){if(F==="highp"){if(s.getShaderPrecisionFormat(s.VERTEX_SHADER,s.HIGH_FLOAT).precision>0&&s.getShaderPrecisionFormat(s.FRAGMENT_SHADER,s.HIGH_FLOAT).precision>0)return"highp";F="mediump"}return F==="mediump"&&s.getShaderPrecisionFormat(s.VERTEX_SHADER,s.MEDIUM_FLOAT).precision>0&&s.getShaderPrecisionFormat(s.FRAGMENT_SHADER,s.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let _=t.precision!==void 0?t.precision:"highp";const M=g(_);M!==_&&(tt("WebGLRenderer:",_,"not supported, using",M,"instead."),_=M);const u=t.logarithmicDepthBuffer===!0,f=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control");t.reversedDepthBuffer===!0&&f===!1&&tt("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");const p=s.getParameter(s.MAX_TEXTURE_IMAGE_UNITS),y=s.getParameter(s.MAX_VERTEX_TEXTURE_IMAGE_UNITS),E=s.getParameter(s.MAX_TEXTURE_SIZE),S=s.getParameter(s.MAX_CUBE_MAP_TEXTURE_SIZE),v=s.getParameter(s.MAX_VERTEX_ATTRIBS),A=s.getParameter(s.MAX_VERTEX_UNIFORM_VECTORS),P=s.getParameter(s.MAX_VARYING_VECTORS),L=s.getParameter(s.MAX_FRAGMENT_UNIFORM_VECTORS),z=s.getParameter(s.MAX_SAMPLES),D=s.getParameter(s.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:l,getMaxPrecision:g,textureFormatReadable:d,textureTypeReadable:m,precision:_,logarithmicDepthBuffer:u,reversedDepthBuffer:f,maxTextures:p,maxVertexTextures:y,maxTextureSize:E,maxCubemapSize:S,maxAttributes:v,maxVertexUniforms:A,maxVaryings:P,maxFragmentUniforms:L,maxSamples:z,samples:D}}function Qy(s){const e=this;let t=null,r=0,a=!1,l=!1;const d=new Jr,m=new lt,g={value:null,needsUpdate:!1};this.uniform=g,this.numPlanes=0,this.numIntersection=0,this.init=function(u,f){const p=u.length!==0||f||r!==0||a;return a=f,r=u.length,p},this.beginShadows=function(){l=!0,M(null)},this.endShadows=function(){l=!1},this.setGlobalState=function(u,f){t=M(u,f,0)},this.setState=function(u,f,p){const y=u.clippingPlanes,E=u.clipIntersection,S=u.clipShadows,v=s.get(u);if(!a||y===null||y.length===0||l&&!S)l?M(null):_();else{const A=l?0:r,P=A*4;let L=v.clippingState||null;g.value=L,L=M(y,f,P,p);for(let z=0;z!==P;++z)L[z]=t[z];v.clippingState=L,this.numIntersection=E?this.numPlanes:0,this.numPlanes+=A}};function _(){g.value!==t&&(g.value=t,g.needsUpdate=r>0),e.numPlanes=r,e.numIntersection=0}function M(u,f,p,y){const E=u!==null?u.length:0;let S=null;if(E!==0){if(S=g.value,y!==!0||S===null){const v=p+E*4,A=f.matrixWorldInverse;m.getNormalMatrix(A),(S===null||S.length<v)&&(S=new Float32Array(v));for(let P=0,L=p;P!==E;++P,L+=4)d.copy(u[P]).applyMatrix4(A,m),d.normal.toArray(S,L),S[L+3]=d.constant}g.value=S,g.needsUpdate=!0}return e.numPlanes=E,e.numIntersection=0,S}}const Lr=4,Um=[.125,.215,.35,.446,.526,.582],ts=20,Jy=256,jo=new O_,Fm=new At;let hf=null,pf=0,mf=0,_f=!1;const eM=new oe;class Om{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,r=.1,a=100,l={}){const{size:d=256,position:m=eM}=l;hf=this._renderer.getRenderTarget(),pf=this._renderer.getActiveCubeFace(),mf=this._renderer.getActiveMipmapLevel(),_f=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(d);const g=this._allocateTargets();return g.depthBuffer=!0,this._sceneToCubeUV(e,r,a,g,m),t>0&&this._blur(g,0,0,t),this._applyPMREM(g),this._cleanup(g),g}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=zm(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=km(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(hf,pf,mf),this._renderer.xr.enabled=_f,e.scissorTest=!1,qs(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===ss||e.mapping===Qs?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),hf=this._renderer.getRenderTarget(),pf=this._renderer.getActiveCubeFace(),mf=this._renderer.getActiveMipmapLevel(),_f=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const r=t||this._allocateTargets();return this._textureToCubeUV(e,r),this._applyPMREM(r),this._cleanup(r),r}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,r={magFilter:Tn,minFilter:Tn,generateMipmaps:!1,type:nr,format:gi,colorSpace:jl,depthBuffer:!1},a=Bm(e,t,r);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Bm(e,t,r);const{_lodMax:l}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=tM(l)),this._blurMaterial=iM(l,e,t),this._ggxMaterial=nM(l,e,t)}return a}_compileMaterial(e){const t=new rr(new ri,e);this._renderer.compile(t,jo)}_sceneToCubeUV(e,t,r,a,l){const g=new ni(90,1,t,r),_=[1,-1,1,1,1,1],M=[1,1,1,-1,-1,-1],u=this._renderer,f=u.autoClear,p=u.toneMapping;u.getClearColor(Fm),u.toneMapping=Di,u.autoClear=!1,u.state.buffers.depth.getReversed()&&(u.setRenderTarget(a),u.clearDepth(),u.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new rr(new la,new P_({name:"PMREM.Background",side:kn,depthWrite:!1,depthTest:!1})));const E=this._backgroundBox,S=E.material;let v=!1;const A=e.background;A?A.isColor&&(S.color.copy(A),e.background=null,v=!0):(S.color.copy(Fm),v=!0);for(let P=0;P<6;P++){const L=P%3;L===0?(g.up.set(0,_[P],0),g.position.set(l.x,l.y,l.z),g.lookAt(l.x+M[P],l.y,l.z)):L===1?(g.up.set(0,0,_[P]),g.position.set(l.x,l.y,l.z),g.lookAt(l.x,l.y+M[P],l.z)):(g.up.set(0,_[P],0),g.position.set(l.x,l.y,l.z),g.lookAt(l.x,l.y,l.z+M[P]));const z=this._cubeSize;qs(a,L*z,P>2?z:0,z,z),u.setRenderTarget(a),v&&u.render(E,g),u.render(e,g)}u.toneMapping=p,u.autoClear=f,e.background=A}_textureToCubeUV(e,t){const r=this._renderer,a=e.mapping===ss||e.mapping===Qs;a?(this._cubemapMaterial===null&&(this._cubemapMaterial=zm()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=km());const l=a?this._cubemapMaterial:this._equirectMaterial,d=this._lodMeshes[0];d.material=l;const m=l.uniforms;m.envMap.value=e;const g=this._cubeSize;qs(t,0,0,3*g,2*g),r.setRenderTarget(t),r.render(d,jo)}_applyPMREM(e){const t=this._renderer,r=t.autoClear;t.autoClear=!1;const a=this._lodMeshes.length;for(let l=1;l<a;l++)this._applyGGXFilter(e,l-1,l);t.autoClear=r}_applyGGXFilter(e,t,r){const a=this._renderer,l=this._pingPongRenderTarget,d=this._ggxMaterial,m=this._lodMeshes[r];m.material=d;const g=d.uniforms,_=r/(this._lodMeshes.length-1),M=t/(this._lodMeshes.length-1),u=Math.sqrt(_*_-M*M),f=0+_*1.25,p=u*f,{_lodMax:y}=this,E=this._sizeLods[r],S=3*E*(r>y-Lr?r-y+Lr:0),v=4*(this._cubeSize-E);g.envMap.value=e.texture,g.roughness.value=p,g.mipInt.value=y-t,qs(l,S,v,3*E,2*E),a.setRenderTarget(l),a.render(m,jo),g.envMap.value=l.texture,g.roughness.value=0,g.mipInt.value=y-r,qs(e,S,v,3*E,2*E),a.setRenderTarget(e),a.render(m,jo)}_blur(e,t,r,a,l){const d=this._pingPongRenderTarget;this._halfBlur(e,d,t,r,a,"latitudinal",l),this._halfBlur(d,e,r,r,a,"longitudinal",l)}_halfBlur(e,t,r,a,l,d,m){const g=this._renderer,_=this._blurMaterial;d!=="latitudinal"&&d!=="longitudinal"&&Mt("blur direction must be either latitudinal or longitudinal!");const M=3,u=this._lodMeshes[a];u.material=_;const f=_.uniforms,p=this._sizeLods[r]-1,y=isFinite(l)?Math.PI/(2*p):2*Math.PI/(2*ts-1),E=l/y,S=isFinite(l)?1+Math.floor(M*E):ts;S>ts&&tt(`sigmaRadians, ${l}, is too large and will clip, as it requested ${S} samples when the maximum is set to ${ts}`);const v=[];let A=0;for(let F=0;F<ts;++F){const R=F/E,I=Math.exp(-R*R/2);v.push(I),F===0?A+=I:F<S&&(A+=2*I)}for(let F=0;F<v.length;F++)v[F]=v[F]/A;f.envMap.value=e.texture,f.samples.value=S,f.weights.value=v,f.latitudinal.value=d==="latitudinal",m&&(f.poleAxis.value=m);const{_lodMax:P}=this;f.dTheta.value=y,f.mipInt.value=P-r;const L=this._sizeLods[a],z=3*L*(a>P-Lr?a-P+Lr:0),D=4*(this._cubeSize-L);qs(t,z,D,3*L,2*L),g.setRenderTarget(t),g.render(u,jo)}}function tM(s){const e=[],t=[],r=[];let a=s;const l=s-Lr+1+Um.length;for(let d=0;d<l;d++){const m=Math.pow(2,a);e.push(m);let g=1/m;d>s-Lr?g=Um[d-s+Lr-1]:d===0&&(g=0),t.push(g);const _=1/(m-2),M=-_,u=1+_,f=[M,M,u,M,u,u,M,M,u,u,M,u],p=6,y=6,E=3,S=2,v=1,A=new Float32Array(E*y*p),P=new Float32Array(S*y*p),L=new Float32Array(v*y*p);for(let D=0;D<p;D++){const F=D%3*2/3-1,R=D>2?0:-1,I=[F,R,0,F+2/3,R,0,F+2/3,R+1,0,F,R,0,F+2/3,R+1,0,F,R+1,0];A.set(I,E*y*D),P.set(f,S*y*D);const W=[D,D,D,D,D,D];L.set(W,v*y*D)}const z=new ri;z.setAttribute("position",new Zt(A,E)),z.setAttribute("uv",new Zt(P,S)),z.setAttribute("faceIndex",new Zt(L,v)),r.push(new rr(z,null)),a>Lr&&a--}return{lodMeshes:r,sizeLods:e,sigmas:t}}function Bm(s,e,t){const r=new Ii(s,e,t);return r.texture.mapping=Jl,r.texture.name="PMREM.cubeUv",r.scissorTest=!0,r}function qs(s,e,t,r,a){s.viewport.set(e,t,r,a),s.scissor.set(e,t,r,a)}function nM(s,e,t){return new xi({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:Jy,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${s}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:nu(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:Ji,depthTest:!1,depthWrite:!1})}function iM(s,e,t){const r=new Float32Array(ts),a=new oe(0,1,0);return new xi({name:"SphericalGaussianBlur",defines:{n:ts,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${s}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:r},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:a}},vertexShader:nu(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Ji,depthTest:!1,depthWrite:!1})}function km(){return new xi({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:nu(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Ji,depthTest:!1,depthWrite:!1})}function zm(){return new xi({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:nu(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Ji,depthTest:!1,depthWrite:!1})}function nu(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}class z_ extends Ii{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const r={width:e,height:e,depth:1},a=[r,r,r,r,r,r];this.texture=new I_(a),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const r={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},a=new la(5,5,5),l=new xi({name:"CubemapFromEquirect",uniforms:eo(r.uniforms),vertexShader:r.vertexShader,fragmentShader:r.fragmentShader,side:kn,blending:Ji});l.uniforms.tEquirect.value=t;const d=new rr(a,l),m=t.minFilter;return t.minFilter===ns&&(t.minFilter=Tn),new ux(1,10,this).update(e,d),t.minFilter=m,d.geometry.dispose(),d.material.dispose(),this}clear(e,t=!0,r=!0,a=!0){const l=e.getRenderTarget();for(let d=0;d<6;d++)e.setRenderTarget(this,d),e.clear(t,r,a);e.setRenderTarget(l)}}function rM(s){let e=new WeakMap,t=new WeakMap,r=null;function a(f,p=!1){return f==null?null:p?d(f):l(f)}function l(f){if(f&&f.isTexture){const p=f.mapping;if(p===Hc||p===Vc)if(e.has(f)){const y=e.get(f).texture;return m(y,f.mapping)}else{const y=f.image;if(y&&y.height>0){const E=new z_(y.height);return E.fromEquirectangularTexture(s,f),e.set(f,E),f.addEventListener("dispose",_),m(E.texture,f.mapping)}else return null}}return f}function d(f){if(f&&f.isTexture){const p=f.mapping,y=p===Hc||p===Vc,E=p===ss||p===Qs;if(y||E){let S=t.get(f);const v=S!==void 0?S.texture.pmremVersion:0;if(f.isRenderTargetTexture&&f.pmremVersion!==v)return r===null&&(r=new Om(s)),S=y?r.fromEquirectangular(f,S):r.fromCubemap(f,S),S.texture.pmremVersion=f.pmremVersion,t.set(f,S),S.texture;if(S!==void 0)return S.texture;{const A=f.image;return y&&A&&A.height>0||E&&A&&g(A)?(r===null&&(r=new Om(s)),S=y?r.fromEquirectangular(f):r.fromCubemap(f),S.texture.pmremVersion=f.pmremVersion,t.set(f,S),f.addEventListener("dispose",M),S.texture):null}}}return f}function m(f,p){return p===Hc?f.mapping=ss:p===Vc&&(f.mapping=Qs),f}function g(f){let p=0;const y=6;for(let E=0;E<y;E++)f[E]!==void 0&&p++;return p===y}function _(f){const p=f.target;p.removeEventListener("dispose",_);const y=e.get(p);y!==void 0&&(e.delete(p),y.dispose())}function M(f){const p=f.target;p.removeEventListener("dispose",M);const y=t.get(p);y!==void 0&&(t.delete(p),y.dispose())}function u(){e=new WeakMap,t=new WeakMap,r!==null&&(r.dispose(),r=null)}return{get:a,dispose:u}}function sM(s){const e={};function t(r){if(e[r]!==void 0)return e[r];const a=s.getExtension(r);return e[r]=a,a}return{has:function(r){return t(r)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(r){const a=t(r);return a===null&&ud("WebGLRenderer: "+r+" extension not supported."),a}}}function oM(s,e,t,r){const a={},l=new WeakMap;function d(u){const f=u.target;f.index!==null&&e.remove(f.index);for(const y in f.attributes)e.remove(f.attributes[y]);f.removeEventListener("dispose",d),delete a[f.id];const p=l.get(f);p&&(e.remove(p),l.delete(f)),r.releaseStatesOfGeometry(f),f.isInstancedBufferGeometry===!0&&delete f._maxInstanceCount,t.memory.geometries--}function m(u,f){return a[f.id]===!0||(f.addEventListener("dispose",d),a[f.id]=!0,t.memory.geometries++),f}function g(u){const f=u.attributes;for(const p in f)e.update(f[p],s.ARRAY_BUFFER)}function _(u){const f=[],p=u.index,y=u.attributes.position;let E=0;if(y===void 0)return;if(p!==null){const A=p.array;E=p.version;for(let P=0,L=A.length;P<L;P+=3){const z=A[P+0],D=A[P+1],F=A[P+2];f.push(z,D,D,F,F,z)}}else{const A=y.array;E=y.version;for(let P=0,L=A.length/3-1;P<L;P+=3){const z=P+0,D=P+1,F=P+2;f.push(z,D,D,F,F,z)}}const S=new(y.count>=65535?C_:R_)(f,1);S.version=E;const v=l.get(u);v&&e.remove(v),l.set(u,S)}function M(u){const f=l.get(u);if(f){const p=u.index;p!==null&&f.version<p.version&&_(u)}else _(u);return l.get(u)}return{get:m,update:g,getWireframeAttribute:M}}function aM(s,e,t){let r;function a(u){r=u}let l,d;function m(u){l=u.type,d=u.bytesPerElement}function g(u,f){s.drawElements(r,f,l,u*d),t.update(f,r,1)}function _(u,f,p){p!==0&&(s.drawElementsInstanced(r,f,l,u*d,p),t.update(f,r,p))}function M(u,f,p){if(p===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(r,f,0,l,u,0,p);let E=0;for(let S=0;S<p;S++)E+=f[S];t.update(E,r,1)}this.setMode=a,this.setIndex=m,this.render=g,this.renderInstances=_,this.renderMultiDraw=M}function lM(s){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function r(l,d,m){switch(t.calls++,d){case s.TRIANGLES:t.triangles+=m*(l/3);break;case s.LINES:t.lines+=m*(l/2);break;case s.LINE_STRIP:t.lines+=m*(l-1);break;case s.LINE_LOOP:t.lines+=m*l;break;case s.POINTS:t.points+=m*l;break;default:Mt("WebGLInfo: Unknown draw mode:",d);break}}function a(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:a,update:r}}function uM(s,e,t){const r=new WeakMap,a=new Jt;function l(d,m,g){const _=d.morphTargetInfluences,M=m.morphAttributes.position||m.morphAttributes.normal||m.morphAttributes.color,u=M!==void 0?M.length:0;let f=r.get(m);if(f===void 0||f.count!==u){let I=function(){F.dispose(),r.delete(m),m.removeEventListener("dispose",I)};f!==void 0&&f.texture.dispose();const p=m.morphAttributes.position!==void 0,y=m.morphAttributes.normal!==void 0,E=m.morphAttributes.color!==void 0,S=m.morphAttributes.position||[],v=m.morphAttributes.normal||[],A=m.morphAttributes.color||[];let P=0;p===!0&&(P=1),y===!0&&(P=2),E===!0&&(P=3);let L=m.attributes.position.count*P,z=1;L>e.maxTextureSize&&(z=Math.ceil(L/e.maxTextureSize),L=e.maxTextureSize);const D=new Float32Array(L*z*4*u),F=new T_(D,L,z,u);F.type=Pi,F.needsUpdate=!0;const R=P*4;for(let W=0;W<u;W++){const O=S[W],j=v[W],re=A[W],ae=L*z*4*W;for(let X=0;X<O.count;X++){const Z=X*R;p===!0&&(a.fromBufferAttribute(O,X),D[ae+Z+0]=a.x,D[ae+Z+1]=a.y,D[ae+Z+2]=a.z,D[ae+Z+3]=0),y===!0&&(a.fromBufferAttribute(j,X),D[ae+Z+4]=a.x,D[ae+Z+5]=a.y,D[ae+Z+6]=a.z,D[ae+Z+7]=0),E===!0&&(a.fromBufferAttribute(re,X),D[ae+Z+8]=a.x,D[ae+Z+9]=a.y,D[ae+Z+10]=a.z,D[ae+Z+11]=re.itemSize===4?a.w:1)}}f={count:u,texture:F,size:new It(L,z)},r.set(m,f),m.addEventListener("dispose",I)}if(d.isInstancedMesh===!0&&d.morphTexture!==null)g.getUniforms().setValue(s,"morphTexture",d.morphTexture,t);else{let p=0;for(let E=0;E<_.length;E++)p+=_[E];const y=m.morphTargetsRelative?1:1-p;g.getUniforms().setValue(s,"morphTargetBaseInfluence",y),g.getUniforms().setValue(s,"morphTargetInfluences",_)}g.getUniforms().setValue(s,"morphTargetsTexture",f.texture,t),g.getUniforms().setValue(s,"morphTargetsTextureSize",f.size)}return{update:l}}function cM(s,e,t,r,a){let l=new WeakMap;function d(_){const M=a.render.frame,u=_.geometry,f=e.get(_,u);if(l.get(f)!==M&&(e.update(f),l.set(f,M)),_.isInstancedMesh&&(_.hasEventListener("dispose",g)===!1&&_.addEventListener("dispose",g),l.get(_)!==M&&(t.update(_.instanceMatrix,s.ARRAY_BUFFER),_.instanceColor!==null&&t.update(_.instanceColor,s.ARRAY_BUFFER),l.set(_,M))),_.isSkinnedMesh){const p=_.skeleton;l.get(p)!==M&&(p.update(),l.set(p,M))}return f}function m(){l=new WeakMap}function g(_){const M=_.target;M.removeEventListener("dispose",g),r.releaseStatesOfObject(M),t.remove(M.instanceMatrix),M.instanceColor!==null&&t.remove(M.instanceColor)}return{update:d,dispose:m}}const fM={[l_]:"LINEAR_TONE_MAPPING",[u_]:"REINHARD_TONE_MAPPING",[c_]:"CINEON_TONE_MAPPING",[f_]:"ACES_FILMIC_TONE_MAPPING",[h_]:"AGX_TONE_MAPPING",[p_]:"NEUTRAL_TONE_MAPPING",[d_]:"CUSTOM_TONE_MAPPING"};function dM(s,e,t,r,a){const l=new Ii(e,t,{type:s,depthBuffer:r,stencilBuffer:a,depthTexture:r?new Js(e,t):void 0}),d=new Ii(e,t,{type:nr,depthBuffer:!1,stencilBuffer:!1}),m=new ri;m.setAttribute("position",new tr([-1,3,0,-1,-1,0,3,-1,0],3)),m.setAttribute("uv",new tr([0,2,0,0,2,0],2));const g=new ox({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),_=new rr(m,g),M=new O_(-1,1,1,-1,0,1);let u=null,f=null,p=!1,y,E=null,S=[],v=!1;this.setSize=function(A,P){l.setSize(A,P),d.setSize(A,P);for(let L=0;L<S.length;L++){const z=S[L];z.setSize&&z.setSize(A,P)}},this.setEffects=function(A){S=A,v=S.length>0&&S[0].isRenderPass===!0;const P=l.width,L=l.height;for(let z=0;z<S.length;z++){const D=S[z];D.setSize&&D.setSize(P,L)}},this.begin=function(A,P){if(p||A.toneMapping===Di&&S.length===0)return!1;if(E=P,P!==null){const L=P.width,z=P.height;(l.width!==L||l.height!==z)&&this.setSize(L,z)}return v===!1&&A.setRenderTarget(l),y=A.toneMapping,A.toneMapping=Di,!0},this.hasRenderPass=function(){return v},this.end=function(A,P){A.toneMapping=y,p=!0;let L=l,z=d;for(let D=0;D<S.length;D++){const F=S[D];if(F.enabled!==!1&&(F.render(A,z,L,P),F.needsSwap!==!1)){const R=L;L=z,z=R}}if(u!==A.outputColorSpace||f!==A.toneMapping){u=A.outputColorSpace,f=A.toneMapping,g.defines={},xt.getTransfer(u)===Lt&&(g.defines.SRGB_TRANSFER="");const D=fM[f];D&&(g.defines[D]=""),g.needsUpdate=!0}g.uniforms.tDiffuse.value=L.texture,A.setRenderTarget(E),A.render(_,M),E=null,p=!1},this.isCompositing=function(){return p},this.dispose=function(){l.depthTexture&&l.depthTexture.dispose(),l.dispose(),d.dispose(),m.dispose(),g.dispose()}}const H_=new Pn,fd=new Js(1,1),V_=new T_,G_=new Ov,W_=new I_,Hm=[],Vm=[],Gm=new Float32Array(16),Wm=new Float32Array(9),Xm=new Float32Array(4);function ro(s,e,t){const r=s[0];if(r<=0||r>0)return s;const a=e*t;let l=Hm[a];if(l===void 0&&(l=new Float32Array(a),Hm[a]=l),e!==0){r.toArray(l,0);for(let d=1,m=0;d!==e;++d)m+=t,s[d].toArray(l,m)}return l}function an(s,e){if(s.length!==e.length)return!1;for(let t=0,r=s.length;t<r;t++)if(s[t]!==e[t])return!1;return!0}function ln(s,e){for(let t=0,r=e.length;t<r;t++)s[t]=e[t]}function iu(s,e){let t=Vm[e];t===void 0&&(t=new Int32Array(e),Vm[e]=t);for(let r=0;r!==e;++r)t[r]=s.allocateTextureUnit();return t}function hM(s,e){const t=this.cache;t[0]!==e&&(s.uniform1f(this.addr,e),t[0]=e)}function pM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(s.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(an(t,e))return;s.uniform2fv(this.addr,e),ln(t,e)}}function mM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(s.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(s.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(an(t,e))return;s.uniform3fv(this.addr,e),ln(t,e)}}function _M(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(s.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(an(t,e))return;s.uniform4fv(this.addr,e),ln(t,e)}}function gM(s,e){const t=this.cache,r=e.elements;if(r===void 0){if(an(t,e))return;s.uniformMatrix2fv(this.addr,!1,e),ln(t,e)}else{if(an(t,r))return;Xm.set(r),s.uniformMatrix2fv(this.addr,!1,Xm),ln(t,r)}}function vM(s,e){const t=this.cache,r=e.elements;if(r===void 0){if(an(t,e))return;s.uniformMatrix3fv(this.addr,!1,e),ln(t,e)}else{if(an(t,r))return;Wm.set(r),s.uniformMatrix3fv(this.addr,!1,Wm),ln(t,r)}}function xM(s,e){const t=this.cache,r=e.elements;if(r===void 0){if(an(t,e))return;s.uniformMatrix4fv(this.addr,!1,e),ln(t,e)}else{if(an(t,r))return;Gm.set(r),s.uniformMatrix4fv(this.addr,!1,Gm),ln(t,r)}}function SM(s,e){const t=this.cache;t[0]!==e&&(s.uniform1i(this.addr,e),t[0]=e)}function yM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(s.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(an(t,e))return;s.uniform2iv(this.addr,e),ln(t,e)}}function MM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(s.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(an(t,e))return;s.uniform3iv(this.addr,e),ln(t,e)}}function EM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(s.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(an(t,e))return;s.uniform4iv(this.addr,e),ln(t,e)}}function TM(s,e){const t=this.cache;t[0]!==e&&(s.uniform1ui(this.addr,e),t[0]=e)}function wM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(s.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(an(t,e))return;s.uniform2uiv(this.addr,e),ln(t,e)}}function AM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(s.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(an(t,e))return;s.uniform3uiv(this.addr,e),ln(t,e)}}function RM(s,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(s.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(an(t,e))return;s.uniform4uiv(this.addr,e),ln(t,e)}}function CM(s,e,t){const r=this.cache,a=t.allocateTextureUnit();r[0]!==a&&(s.uniform1i(this.addr,a),r[0]=a);let l;this.type===s.SAMPLER_2D_SHADOW?(fd.compareFunction=t.isReversedDepthBuffer()?Md:yd,l=fd):l=H_,t.setTexture2D(e||l,a)}function bM(s,e,t){const r=this.cache,a=t.allocateTextureUnit();r[0]!==a&&(s.uniform1i(this.addr,a),r[0]=a),t.setTexture3D(e||G_,a)}function PM(s,e,t){const r=this.cache,a=t.allocateTextureUnit();r[0]!==a&&(s.uniform1i(this.addr,a),r[0]=a),t.setTextureCube(e||W_,a)}function LM(s,e,t){const r=this.cache,a=t.allocateTextureUnit();r[0]!==a&&(s.uniform1i(this.addr,a),r[0]=a),t.setTexture2DArray(e||V_,a)}function DM(s){switch(s){case 5126:return hM;case 35664:return pM;case 35665:return mM;case 35666:return _M;case 35674:return gM;case 35675:return vM;case 35676:return xM;case 5124:case 35670:return SM;case 35667:case 35671:return yM;case 35668:case 35672:return MM;case 35669:case 35673:return EM;case 5125:return TM;case 36294:return wM;case 36295:return AM;case 36296:return RM;case 35678:case 36198:case 36298:case 36306:case 35682:return CM;case 35679:case 36299:case 36307:return bM;case 35680:case 36300:case 36308:case 36293:return PM;case 36289:case 36303:case 36311:case 36292:return LM}}function IM(s,e){s.uniform1fv(this.addr,e)}function NM(s,e){const t=ro(e,this.size,2);s.uniform2fv(this.addr,t)}function UM(s,e){const t=ro(e,this.size,3);s.uniform3fv(this.addr,t)}function FM(s,e){const t=ro(e,this.size,4);s.uniform4fv(this.addr,t)}function OM(s,e){const t=ro(e,this.size,4);s.uniformMatrix2fv(this.addr,!1,t)}function BM(s,e){const t=ro(e,this.size,9);s.uniformMatrix3fv(this.addr,!1,t)}function kM(s,e){const t=ro(e,this.size,16);s.uniformMatrix4fv(this.addr,!1,t)}function zM(s,e){s.uniform1iv(this.addr,e)}function HM(s,e){s.uniform2iv(this.addr,e)}function VM(s,e){s.uniform3iv(this.addr,e)}function GM(s,e){s.uniform4iv(this.addr,e)}function WM(s,e){s.uniform1uiv(this.addr,e)}function XM(s,e){s.uniform2uiv(this.addr,e)}function YM(s,e){s.uniform3uiv(this.addr,e)}function qM(s,e){s.uniform4uiv(this.addr,e)}function jM(s,e,t){const r=this.cache,a=e.length,l=iu(t,a);an(r,l)||(s.uniform1iv(this.addr,l),ln(r,l));let d;this.type===s.SAMPLER_2D_SHADOW?d=fd:d=H_;for(let m=0;m!==a;++m)t.setTexture2D(e[m]||d,l[m])}function KM(s,e,t){const r=this.cache,a=e.length,l=iu(t,a);an(r,l)||(s.uniform1iv(this.addr,l),ln(r,l));for(let d=0;d!==a;++d)t.setTexture3D(e[d]||G_,l[d])}function $M(s,e,t){const r=this.cache,a=e.length,l=iu(t,a);an(r,l)||(s.uniform1iv(this.addr,l),ln(r,l));for(let d=0;d!==a;++d)t.setTextureCube(e[d]||W_,l[d])}function ZM(s,e,t){const r=this.cache,a=e.length,l=iu(t,a);an(r,l)||(s.uniform1iv(this.addr,l),ln(r,l));for(let d=0;d!==a;++d)t.setTexture2DArray(e[d]||V_,l[d])}function QM(s){switch(s){case 5126:return IM;case 35664:return NM;case 35665:return UM;case 35666:return FM;case 35674:return OM;case 35675:return BM;case 35676:return kM;case 5124:case 35670:return zM;case 35667:case 35671:return HM;case 35668:case 35672:return VM;case 35669:case 35673:return GM;case 5125:return WM;case 36294:return XM;case 36295:return YM;case 36296:return qM;case 35678:case 36198:case 36298:case 36306:case 35682:return jM;case 35679:case 36299:case 36307:return KM;case 35680:case 36300:case 36308:case 36293:return $M;case 36289:case 36303:case 36311:case 36292:return ZM}}class JM{constructor(e,t,r){this.id=e,this.addr=r,this.cache=[],this.type=t.type,this.setValue=DM(t.type)}}class eE{constructor(e,t,r){this.id=e,this.addr=r,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=QM(t.type)}}class tE{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,r){const a=this.seq;for(let l=0,d=a.length;l!==d;++l){const m=a[l];m.setValue(e,t[m.id],r)}}}const gf=/(\w+)(\])?(\[|\.)?/g;function Ym(s,e){s.seq.push(e),s.map[e.id]=e}function nE(s,e,t){const r=s.name,a=r.length;for(gf.lastIndex=0;;){const l=gf.exec(r),d=gf.lastIndex;let m=l[1];const g=l[2]==="]",_=l[3];if(g&&(m=m|0),_===void 0||_==="["&&d+2===a){Ym(t,_===void 0?new JM(m,s,e):new eE(m,s,e));break}else{let u=t.map[m];u===void 0&&(u=new tE(m),Ym(t,u)),t=u}}}class Wl{constructor(e,t){this.seq=[],this.map={};const r=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let d=0;d<r;++d){const m=e.getActiveUniform(t,d),g=e.getUniformLocation(t,m.name);nE(m,g,this)}const a=[],l=[];for(const d of this.seq)d.type===e.SAMPLER_2D_SHADOW||d.type===e.SAMPLER_CUBE_SHADOW||d.type===e.SAMPLER_2D_ARRAY_SHADOW?a.push(d):l.push(d);a.length>0&&(this.seq=a.concat(l))}setValue(e,t,r,a){const l=this.map[t];l!==void 0&&l.setValue(e,r,a)}setOptional(e,t,r){const a=t[r];a!==void 0&&this.setValue(e,r,a)}static upload(e,t,r,a){for(let l=0,d=t.length;l!==d;++l){const m=t[l],g=r[m.id];g.needsUpdate!==!1&&m.setValue(e,g.value,a)}}static seqWithValue(e,t){const r=[];for(let a=0,l=e.length;a!==l;++a){const d=e[a];d.id in t&&r.push(d)}return r}}function qm(s,e,t){const r=s.createShader(e);return s.shaderSource(r,t),s.compileShader(r),r}const iE=37297;let rE=0;function sE(s,e){const t=s.split(`
`),r=[],a=Math.max(e-6,0),l=Math.min(e+6,t.length);for(let d=a;d<l;d++){const m=d+1;r.push(`${m===e?">":" "} ${m}: ${t[d]}`)}return r.join(`
`)}const jm=new lt;function oE(s){xt._getMatrix(jm,xt.workingColorSpace,s);const e=`mat3( ${jm.elements.map(t=>t.toFixed(4))} )`;switch(xt.getTransfer(s)){case Kl:return[e,"LinearTransferOETF"];case Lt:return[e,"sRGBTransferOETF"];default:return tt("WebGLProgram: Unsupported color space: ",s),[e,"LinearTransferOETF"]}}function Km(s,e,t){const r=s.getShaderParameter(e,s.COMPILE_STATUS),l=(s.getShaderInfoLog(e)||"").trim();if(r&&l==="")return"";const d=/ERROR: 0:(\d+)/.exec(l);if(d){const m=parseInt(d[1]);return t.toUpperCase()+`

`+l+`

`+sE(s.getShaderSource(e),m)}else return l}function aE(s,e){const t=oE(e);return[`vec4 ${s}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}const lE={[l_]:"Linear",[u_]:"Reinhard",[c_]:"Cineon",[f_]:"ACESFilmic",[h_]:"AgX",[p_]:"Neutral",[d_]:"Custom"};function uE(s,e){const t=lE[e];return t===void 0?(tt("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+s+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+s+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Fl=new oe;function cE(){xt.getLuminanceCoefficients(Fl);const s=Fl.x.toFixed(4),e=Fl.y.toFixed(4),t=Fl.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${s}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function fE(s){return[s.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",s.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Qo).join(`
`)}function dE(s){const e=[];for(const t in s){const r=s[t];r!==!1&&e.push("#define "+t+" "+r)}return e.join(`
`)}function hE(s,e){const t={},r=s.getProgramParameter(e,s.ACTIVE_ATTRIBUTES);for(let a=0;a<r;a++){const l=s.getActiveAttrib(e,a),d=l.name;let m=1;l.type===s.FLOAT_MAT2&&(m=2),l.type===s.FLOAT_MAT3&&(m=3),l.type===s.FLOAT_MAT4&&(m=4),t[d]={type:l.type,location:s.getAttribLocation(e,d),locationSize:m}}return t}function Qo(s){return s!==""}function $m(s,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return s.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function Zm(s,e){return s.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const pE=/^[ \t]*#include +<([\w\d./]+)>/gm;function dd(s){return s.replace(pE,_E)}const mE=new Map;function _E(s,e){let t=ht[e];if(t===void 0){const r=mE.get(e);if(r!==void 0)t=ht[r],tt('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,r);else throw new Error("Can not resolve #include <"+e+">")}return dd(t)}const gE=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Qm(s){return s.replace(gE,vE)}function vE(s,e,t,r){let a="";for(let l=parseInt(e);l<parseInt(t);l++)a+=r.replace(/\[\s*i\s*\]/g,"[ "+l+" ]").replace(/UNROLLED_LOOP_INDEX/g,l);return a}function Jm(s){let e=`precision ${s.precision} float;
	precision ${s.precision} int;
	precision ${s.precision} sampler2D;
	precision ${s.precision} samplerCube;
	precision ${s.precision} sampler3D;
	precision ${s.precision} sampler2DArray;
	precision ${s.precision} sampler2DShadow;
	precision ${s.precision} samplerCubeShadow;
	precision ${s.precision} sampler2DArrayShadow;
	precision ${s.precision} isampler2D;
	precision ${s.precision} isampler3D;
	precision ${s.precision} isamplerCube;
	precision ${s.precision} isampler2DArray;
	precision ${s.precision} usampler2D;
	precision ${s.precision} usampler3D;
	precision ${s.precision} usamplerCube;
	precision ${s.precision} usampler2DArray;
	`;return s.precision==="highp"?e+=`
#define HIGH_PRECISION`:s.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:s.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}const xE={[kl]:"SHADOWMAP_TYPE_PCF",[Zo]:"SHADOWMAP_TYPE_VSM"};function SE(s){return xE[s.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}const yE={[ss]:"ENVMAP_TYPE_CUBE",[Qs]:"ENVMAP_TYPE_CUBE",[Jl]:"ENVMAP_TYPE_CUBE_UV"};function ME(s){return s.envMap===!1?"ENVMAP_TYPE_CUBE":yE[s.envMapMode]||"ENVMAP_TYPE_CUBE"}const EE={[Qs]:"ENVMAP_MODE_REFRACTION"};function TE(s){return s.envMap===!1?"ENVMAP_MODE_REFLECTION":EE[s.envMapMode]||"ENVMAP_MODE_REFLECTION"}const wE={[a_]:"ENVMAP_BLENDING_MULTIPLY",[ev]:"ENVMAP_BLENDING_MIX",[tv]:"ENVMAP_BLENDING_ADD"};function AE(s){return s.envMap===!1?"ENVMAP_BLENDING_NONE":wE[s.combine]||"ENVMAP_BLENDING_NONE"}function RE(s){const e=s.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,r=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:r,maxMip:t}}function CE(s,e,t,r){const a=s.getContext(),l=t.defines;let d=t.vertexShader,m=t.fragmentShader;const g=SE(t),_=ME(t),M=TE(t),u=AE(t),f=RE(t),p=fE(t),y=dE(l),E=a.createProgram();let S,v,A=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(S=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y].filter(Qo).join(`
`),S.length>0&&(S+=`
`),v=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y].filter(Qo).join(`
`),v.length>0&&(v+=`
`)):(S=[Jm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+M:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexNormals?"#define HAS_NORMAL":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+g:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Qo).join(`
`),v=[Jm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+_:"",t.envMap?"#define "+M:"",t.envMap?"#define "+u:"",f?"#define CUBEUV_TEXEL_WIDTH "+f.texelWidth:"",f?"#define CUBEUV_TEXEL_HEIGHT "+f.texelHeight:"",f?"#define CUBEUV_MAX_MIP "+f.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor?"#define USE_COLOR":"",t.vertexAlphas||t.batchingColor?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+g:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==Di?"#define TONE_MAPPING":"",t.toneMapping!==Di?ht.tonemapping_pars_fragment:"",t.toneMapping!==Di?uE("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",ht.colorspace_pars_fragment,aE("linearToOutputTexel",t.outputColorSpace),cE(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Qo).join(`
`)),d=dd(d),d=$m(d,t),d=Zm(d,t),m=dd(m),m=$m(m,t),m=Zm(m,t),d=Qm(d),m=Qm(m),t.isRawShaderMaterial!==!0&&(A=`#version 300 es
`,S=[p,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+S,v=["#define varying in",t.glslVersion===um?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===um?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+v);const P=A+S+d,L=A+v+m,z=qm(a,a.VERTEX_SHADER,P),D=qm(a,a.FRAGMENT_SHADER,L);a.attachShader(E,z),a.attachShader(E,D),t.index0AttributeName!==void 0?a.bindAttribLocation(E,0,t.index0AttributeName):t.morphTargets===!0&&a.bindAttribLocation(E,0,"position"),a.linkProgram(E);function F(O){if(s.debug.checkShaderErrors){const j=a.getProgramInfoLog(E)||"",re=a.getShaderInfoLog(z)||"",ae=a.getShaderInfoLog(D)||"",X=j.trim(),Z=re.trim(),q=ae.trim();let G=!0,J=!0;if(a.getProgramParameter(E,a.LINK_STATUS)===!1)if(G=!1,typeof s.debug.onShaderError=="function")s.debug.onShaderError(a,E,z,D);else{const ie=Km(a,z,"vertex"),U=Km(a,D,"fragment");Mt("THREE.WebGLProgram: Shader Error "+a.getError()+" - VALIDATE_STATUS "+a.getProgramParameter(E,a.VALIDATE_STATUS)+`

Material Name: `+O.name+`
Material Type: `+O.type+`

Program Info Log: `+X+`
`+ie+`
`+U)}else X!==""?tt("WebGLProgram: Program Info Log:",X):(Z===""||q==="")&&(J=!1);J&&(O.diagnostics={runnable:G,programLog:X,vertexShader:{log:Z,prefix:S},fragmentShader:{log:q,prefix:v}})}a.deleteShader(z),a.deleteShader(D),R=new Wl(a,E),I=hE(a,E)}let R;this.getUniforms=function(){return R===void 0&&F(this),R};let I;this.getAttributes=function(){return I===void 0&&F(this),I};let W=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return W===!1&&(W=a.getProgramParameter(E,iE)),W},this.destroy=function(){r.releaseStatesOfProgram(this),a.deleteProgram(E),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=rE++,this.cacheKey=e,this.usedTimes=1,this.program=E,this.vertexShader=z,this.fragmentShader=D,this}let bE=0;class PE{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,r=e.fragmentShader,a=this._getShaderStage(t),l=this._getShaderStage(r),d=this._getShaderCacheForMaterial(e);return d.has(a)===!1&&(d.add(a),a.usedTimes++),d.has(l)===!1&&(d.add(l),l.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const r of t)r.usedTimes--,r.usedTimes===0&&this.shaderCache.delete(r.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let r=t.get(e);return r===void 0&&(r=new Set,t.set(e,r)),r}_getShaderStage(e){const t=this.shaderCache;let r=t.get(e);return r===void 0&&(r=new LE(e),t.set(e,r)),r}}class LE{constructor(e){this.id=bE++,this.code=e,this.usedTimes=0}}function DE(s){return s===os||s===Yl||s===ql}function IE(s,e,t,r,a,l){const d=new w_,m=new PE,g=new Set,_=[],M=new Map,u=r.logarithmicDepthBuffer;let f=r.precision;const p={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function y(R){return g.add(R),R===0?"uv":`uv${R}`}function E(R,I,W,O,j,re){const ae=O.fog,X=j.geometry,Z=R.isMeshStandardMaterial||R.isMeshLambertMaterial||R.isMeshPhongMaterial?O.environment:null,q=R.isMeshStandardMaterial||R.isMeshLambertMaterial&&!R.envMap||R.isMeshPhongMaterial&&!R.envMap,G=e.get(R.envMap||Z,q),J=G&&G.mapping===Jl?G.image.height:null,ie=p[R.type];R.precision!==null&&(f=r.getMaxPrecision(R.precision),f!==R.precision&&tt("WebGLProgram.getParameters:",R.precision,"not supported, using",f,"instead."));const U=X.morphAttributes.position||X.morphAttributes.normal||X.morphAttributes.color,K=U!==void 0?U.length:0;let Le=0;X.morphAttributes.position!==void 0&&(Le=1),X.morphAttributes.normal!==void 0&&(Le=2),X.morphAttributes.color!==void 0&&(Le=3);let De,we,se,_e;if(ie){const st=bi[ie];De=st.vertexShader,we=st.fragmentShader}else De=R.vertexShader,we=R.fragmentShader,m.update(R),se=m.getVertexShaderID(R),_e=m.getFragmentShaderID(R);const de=s.getRenderTarget(),Ie=s.state.buffers.depth.getReversed(),je=j.isInstancedMesh===!0,$e=j.isBatchedMesh===!0,Ut=!!R.map,ct=!!R.matcap,Et=!!G,Dt=!!R.aoMap,ft=!!R.lightMap,Yt=!!R.bumpMap,Ft=!!R.normalMap,hn=!!R.displacementMap,H=!!R.emissiveMap,Ot=!!R.metalnessMap,dt=!!R.roughnessMap,Ct=R.anisotropy>0,Ne=R.clearcoat>0,zt=R.dispersion>0,b=R.iridescence>0,T=R.sheen>0,$=R.transmission>0,he=Ct&&!!R.anisotropyMap,me=Ne&&!!R.clearcoatMap,ye=Ne&&!!R.clearcoatNormalMap,Pe=Ne&&!!R.clearcoatRoughnessMap,ce=b&&!!R.iridescenceMap,pe=b&&!!R.iridescenceThicknessMap,Fe=T&&!!R.sheenColorMap,Be=T&&!!R.sheenRoughnessMap,Ae=!!R.specularMap,Me=!!R.specularColorMap,et=!!R.specularIntensityMap,rt=$&&!!R.transmissionMap,pt=$&&!!R.thicknessMap,k=!!R.gradientMap,Te=!!R.alphaMap,fe=R.alphaTest>0,Oe=!!R.alphaHash,Ce=!!R.extensions;let ge=Di;R.toneMapped&&(de===null||de.isXRRenderTarget===!0)&&(ge=s.toneMapping);const We={shaderID:ie,shaderType:R.type,shaderName:R.name,vertexShader:De,fragmentShader:we,defines:R.defines,customVertexShaderID:se,customFragmentShaderID:_e,isRawShaderMaterial:R.isRawShaderMaterial===!0,glslVersion:R.glslVersion,precision:f,batching:$e,batchingColor:$e&&j._colorsTexture!==null,instancing:je,instancingColor:je&&j.instanceColor!==null,instancingMorph:je&&j.morphTexture!==null,outputColorSpace:de===null?s.outputColorSpace:de.isXRRenderTarget===!0?de.texture.colorSpace:xt.workingColorSpace,alphaToCoverage:!!R.alphaToCoverage,map:Ut,matcap:ct,envMap:Et,envMapMode:Et&&G.mapping,envMapCubeUVHeight:J,aoMap:Dt,lightMap:ft,bumpMap:Yt,normalMap:Ft,displacementMap:hn,emissiveMap:H,normalMapObjectSpace:Ft&&R.normalMapType===rv,normalMapTangentSpace:Ft&&R.normalMapType===om,packedNormalMap:Ft&&R.normalMapType===om&&DE(R.normalMap.format),metalnessMap:Ot,roughnessMap:dt,anisotropy:Ct,anisotropyMap:he,clearcoat:Ne,clearcoatMap:me,clearcoatNormalMap:ye,clearcoatRoughnessMap:Pe,dispersion:zt,iridescence:b,iridescenceMap:ce,iridescenceThicknessMap:pe,sheen:T,sheenColorMap:Fe,sheenRoughnessMap:Be,specularMap:Ae,specularColorMap:Me,specularIntensityMap:et,transmission:$,transmissionMap:rt,thicknessMap:pt,gradientMap:k,opaque:R.transparent===!1&&R.blending===Ks&&R.alphaToCoverage===!1,alphaMap:Te,alphaTest:fe,alphaHash:Oe,combine:R.combine,mapUv:Ut&&y(R.map.channel),aoMapUv:Dt&&y(R.aoMap.channel),lightMapUv:ft&&y(R.lightMap.channel),bumpMapUv:Yt&&y(R.bumpMap.channel),normalMapUv:Ft&&y(R.normalMap.channel),displacementMapUv:hn&&y(R.displacementMap.channel),emissiveMapUv:H&&y(R.emissiveMap.channel),metalnessMapUv:Ot&&y(R.metalnessMap.channel),roughnessMapUv:dt&&y(R.roughnessMap.channel),anisotropyMapUv:he&&y(R.anisotropyMap.channel),clearcoatMapUv:me&&y(R.clearcoatMap.channel),clearcoatNormalMapUv:ye&&y(R.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Pe&&y(R.clearcoatRoughnessMap.channel),iridescenceMapUv:ce&&y(R.iridescenceMap.channel),iridescenceThicknessMapUv:pe&&y(R.iridescenceThicknessMap.channel),sheenColorMapUv:Fe&&y(R.sheenColorMap.channel),sheenRoughnessMapUv:Be&&y(R.sheenRoughnessMap.channel),specularMapUv:Ae&&y(R.specularMap.channel),specularColorMapUv:Me&&y(R.specularColorMap.channel),specularIntensityMapUv:et&&y(R.specularIntensityMap.channel),transmissionMapUv:rt&&y(R.transmissionMap.channel),thicknessMapUv:pt&&y(R.thicknessMap.channel),alphaMapUv:Te&&y(R.alphaMap.channel),vertexTangents:!!X.attributes.tangent&&(Ft||Ct),vertexNormals:!!X.attributes.normal,vertexColors:R.vertexColors,vertexAlphas:R.vertexColors===!0&&!!X.attributes.color&&X.attributes.color.itemSize===4,pointsUvs:j.isPoints===!0&&!!X.attributes.uv&&(Ut||Te),fog:!!ae,useFog:R.fog===!0,fogExp2:!!ae&&ae.isFogExp2,flatShading:R.wireframe===!1&&(R.flatShading===!0||X.attributes.normal===void 0&&Ft===!1&&(R.isMeshLambertMaterial||R.isMeshPhongMaterial||R.isMeshStandardMaterial||R.isMeshPhysicalMaterial)),sizeAttenuation:R.sizeAttenuation===!0,logarithmicDepthBuffer:u,reversedDepthBuffer:Ie,skinning:j.isSkinnedMesh===!0,morphTargets:X.morphAttributes.position!==void 0,morphNormals:X.morphAttributes.normal!==void 0,morphColors:X.morphAttributes.color!==void 0,morphTargetsCount:K,morphTextureStride:Le,numDirLights:I.directional.length,numPointLights:I.point.length,numSpotLights:I.spot.length,numSpotLightMaps:I.spotLightMap.length,numRectAreaLights:I.rectArea.length,numHemiLights:I.hemi.length,numDirLightShadows:I.directionalShadowMap.length,numPointLightShadows:I.pointShadowMap.length,numSpotLightShadows:I.spotShadowMap.length,numSpotLightShadowsWithMaps:I.numSpotLightShadowsWithMaps,numLightProbes:I.numLightProbes,numLightProbeGrids:re.length,numClippingPlanes:l.numPlanes,numClipIntersection:l.numIntersection,dithering:R.dithering,shadowMapEnabled:s.shadowMap.enabled&&W.length>0,shadowMapType:s.shadowMap.type,toneMapping:ge,decodeVideoTexture:Ut&&R.map.isVideoTexture===!0&&xt.getTransfer(R.map.colorSpace)===Lt,decodeVideoTextureEmissive:H&&R.emissiveMap.isVideoTexture===!0&&xt.getTransfer(R.emissiveMap.colorSpace)===Lt,premultipliedAlpha:R.premultipliedAlpha,doubleSided:R.side===Zi,flipSided:R.side===kn,useDepthPacking:R.depthPacking>=0,depthPacking:R.depthPacking||0,index0AttributeName:R.index0AttributeName,extensionClipCullDistance:Ce&&R.extensions.clipCullDistance===!0&&t.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ce&&R.extensions.multiDraw===!0||$e)&&t.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:t.has("KHR_parallel_shader_compile"),customProgramCacheKey:R.customProgramCacheKey()};return We.vertexUv1s=g.has(1),We.vertexUv2s=g.has(2),We.vertexUv3s=g.has(3),g.clear(),We}function S(R){const I=[];if(R.shaderID?I.push(R.shaderID):(I.push(R.customVertexShaderID),I.push(R.customFragmentShaderID)),R.defines!==void 0)for(const W in R.defines)I.push(W),I.push(R.defines[W]);return R.isRawShaderMaterial===!1&&(v(I,R),A(I,R),I.push(s.outputColorSpace)),I.push(R.customProgramCacheKey),I.join()}function v(R,I){R.push(I.precision),R.push(I.outputColorSpace),R.push(I.envMapMode),R.push(I.envMapCubeUVHeight),R.push(I.mapUv),R.push(I.alphaMapUv),R.push(I.lightMapUv),R.push(I.aoMapUv),R.push(I.bumpMapUv),R.push(I.normalMapUv),R.push(I.displacementMapUv),R.push(I.emissiveMapUv),R.push(I.metalnessMapUv),R.push(I.roughnessMapUv),R.push(I.anisotropyMapUv),R.push(I.clearcoatMapUv),R.push(I.clearcoatNormalMapUv),R.push(I.clearcoatRoughnessMapUv),R.push(I.iridescenceMapUv),R.push(I.iridescenceThicknessMapUv),R.push(I.sheenColorMapUv),R.push(I.sheenRoughnessMapUv),R.push(I.specularMapUv),R.push(I.specularColorMapUv),R.push(I.specularIntensityMapUv),R.push(I.transmissionMapUv),R.push(I.thicknessMapUv),R.push(I.combine),R.push(I.fogExp2),R.push(I.sizeAttenuation),R.push(I.morphTargetsCount),R.push(I.morphAttributeCount),R.push(I.numDirLights),R.push(I.numPointLights),R.push(I.numSpotLights),R.push(I.numSpotLightMaps),R.push(I.numHemiLights),R.push(I.numRectAreaLights),R.push(I.numDirLightShadows),R.push(I.numPointLightShadows),R.push(I.numSpotLightShadows),R.push(I.numSpotLightShadowsWithMaps),R.push(I.numLightProbes),R.push(I.shadowMapType),R.push(I.toneMapping),R.push(I.numClippingPlanes),R.push(I.numClipIntersection),R.push(I.depthPacking)}function A(R,I){d.disableAll(),I.instancing&&d.enable(0),I.instancingColor&&d.enable(1),I.instancingMorph&&d.enable(2),I.matcap&&d.enable(3),I.envMap&&d.enable(4),I.normalMapObjectSpace&&d.enable(5),I.normalMapTangentSpace&&d.enable(6),I.clearcoat&&d.enable(7),I.iridescence&&d.enable(8),I.alphaTest&&d.enable(9),I.vertexColors&&d.enable(10),I.vertexAlphas&&d.enable(11),I.vertexUv1s&&d.enable(12),I.vertexUv2s&&d.enable(13),I.vertexUv3s&&d.enable(14),I.vertexTangents&&d.enable(15),I.anisotropy&&d.enable(16),I.alphaHash&&d.enable(17),I.batching&&d.enable(18),I.dispersion&&d.enable(19),I.batchingColor&&d.enable(20),I.gradientMap&&d.enable(21),I.packedNormalMap&&d.enable(22),I.vertexNormals&&d.enable(23),R.push(d.mask),d.disableAll(),I.fog&&d.enable(0),I.useFog&&d.enable(1),I.flatShading&&d.enable(2),I.logarithmicDepthBuffer&&d.enable(3),I.reversedDepthBuffer&&d.enable(4),I.skinning&&d.enable(5),I.morphTargets&&d.enable(6),I.morphNormals&&d.enable(7),I.morphColors&&d.enable(8),I.premultipliedAlpha&&d.enable(9),I.shadowMapEnabled&&d.enable(10),I.doubleSided&&d.enable(11),I.flipSided&&d.enable(12),I.useDepthPacking&&d.enable(13),I.dithering&&d.enable(14),I.transmission&&d.enable(15),I.sheen&&d.enable(16),I.opaque&&d.enable(17),I.pointsUvs&&d.enable(18),I.decodeVideoTexture&&d.enable(19),I.decodeVideoTextureEmissive&&d.enable(20),I.alphaToCoverage&&d.enable(21),I.numLightProbeGrids>0&&d.enable(22),R.push(d.mask)}function P(R){const I=p[R.type];let W;if(I){const O=bi[I];W=ix.clone(O.uniforms)}else W=R.uniforms;return W}function L(R,I){let W=M.get(I);return W!==void 0?++W.usedTimes:(W=new CE(s,I,R,a),_.push(W),M.set(I,W)),W}function z(R){if(--R.usedTimes===0){const I=_.indexOf(R);_[I]=_[_.length-1],_.pop(),M.delete(R.cacheKey),R.destroy()}}function D(R){m.remove(R)}function F(){m.dispose()}return{getParameters:E,getProgramCacheKey:S,getUniforms:P,acquireProgram:L,releaseProgram:z,releaseShaderCache:D,programs:_,dispose:F}}function NE(){let s=new WeakMap;function e(d){return s.has(d)}function t(d){let m=s.get(d);return m===void 0&&(m={},s.set(d,m)),m}function r(d){s.delete(d)}function a(d,m,g){s.get(d)[m]=g}function l(){s=new WeakMap}return{has:e,get:t,remove:r,update:a,dispose:l}}function UE(s,e){return s.groupOrder!==e.groupOrder?s.groupOrder-e.groupOrder:s.renderOrder!==e.renderOrder?s.renderOrder-e.renderOrder:s.material.id!==e.material.id?s.material.id-e.material.id:s.materialVariant!==e.materialVariant?s.materialVariant-e.materialVariant:s.z!==e.z?s.z-e.z:s.id-e.id}function e_(s,e){return s.groupOrder!==e.groupOrder?s.groupOrder-e.groupOrder:s.renderOrder!==e.renderOrder?s.renderOrder-e.renderOrder:s.z!==e.z?e.z-s.z:s.id-e.id}function t_(){const s=[];let e=0;const t=[],r=[],a=[];function l(){e=0,t.length=0,r.length=0,a.length=0}function d(f){let p=0;return f.isInstancedMesh&&(p+=2),f.isSkinnedMesh&&(p+=1),p}function m(f,p,y,E,S,v){let A=s[e];return A===void 0?(A={id:f.id,object:f,geometry:p,material:y,materialVariant:d(f),groupOrder:E,renderOrder:f.renderOrder,z:S,group:v},s[e]=A):(A.id=f.id,A.object=f,A.geometry=p,A.material=y,A.materialVariant=d(f),A.groupOrder=E,A.renderOrder=f.renderOrder,A.z=S,A.group=v),e++,A}function g(f,p,y,E,S,v){const A=m(f,p,y,E,S,v);y.transmission>0?r.push(A):y.transparent===!0?a.push(A):t.push(A)}function _(f,p,y,E,S,v){const A=m(f,p,y,E,S,v);y.transmission>0?r.unshift(A):y.transparent===!0?a.unshift(A):t.unshift(A)}function M(f,p){t.length>1&&t.sort(f||UE),r.length>1&&r.sort(p||e_),a.length>1&&a.sort(p||e_)}function u(){for(let f=e,p=s.length;f<p;f++){const y=s[f];if(y.id===null)break;y.id=null,y.object=null,y.geometry=null,y.material=null,y.group=null}}return{opaque:t,transmissive:r,transparent:a,init:l,push:g,unshift:_,finish:u,sort:M}}function FE(){let s=new WeakMap;function e(r,a){const l=s.get(r);let d;return l===void 0?(d=new t_,s.set(r,[d])):a>=l.length?(d=new t_,l.push(d)):d=l[a],d}function t(){s=new WeakMap}return{get:e,dispose:t}}function OE(){const s={};return{get:function(e){if(s[e.id]!==void 0)return s[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new oe,color:new At};break;case"SpotLight":t={position:new oe,direction:new oe,color:new At,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new oe,color:new At,distance:0,decay:0};break;case"HemisphereLight":t={direction:new oe,skyColor:new At,groundColor:new At};break;case"RectAreaLight":t={color:new At,position:new oe,halfWidth:new oe,halfHeight:new oe};break}return s[e.id]=t,t}}}function BE(){const s={};return{get:function(e){if(s[e.id]!==void 0)return s[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new It};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new It};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new It,shadowCameraNear:1,shadowCameraFar:1e3};break}return s[e.id]=t,t}}}let kE=0;function zE(s,e){return(e.castShadow?2:0)-(s.castShadow?2:0)+(e.map?1:0)-(s.map?1:0)}function HE(s){const e=new OE,t=BE(),r={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let _=0;_<9;_++)r.probe.push(new oe);const a=new oe,l=new rn,d=new rn;function m(_){let M=0,u=0,f=0;for(let I=0;I<9;I++)r.probe[I].set(0,0,0);let p=0,y=0,E=0,S=0,v=0,A=0,P=0,L=0,z=0,D=0,F=0;_.sort(zE);for(let I=0,W=_.length;I<W;I++){const O=_[I],j=O.color,re=O.intensity,ae=O.distance;let X=null;if(O.shadow&&O.shadow.map&&(O.shadow.map.texture.format===os?X=O.shadow.map.texture:X=O.shadow.map.depthTexture||O.shadow.map.texture),O.isAmbientLight)M+=j.r*re,u+=j.g*re,f+=j.b*re;else if(O.isLightProbe){for(let Z=0;Z<9;Z++)r.probe[Z].addScaledVector(O.sh.coefficients[Z],re);F++}else if(O.isDirectionalLight){const Z=e.get(O);if(Z.color.copy(O.color).multiplyScalar(O.intensity),O.castShadow){const q=O.shadow,G=t.get(O);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,r.directionalShadow[p]=G,r.directionalShadowMap[p]=X,r.directionalShadowMatrix[p]=O.shadow.matrix,A++}r.directional[p]=Z,p++}else if(O.isSpotLight){const Z=e.get(O);Z.position.setFromMatrixPosition(O.matrixWorld),Z.color.copy(j).multiplyScalar(re),Z.distance=ae,Z.coneCos=Math.cos(O.angle),Z.penumbraCos=Math.cos(O.angle*(1-O.penumbra)),Z.decay=O.decay,r.spot[E]=Z;const q=O.shadow;if(O.map&&(r.spotLightMap[z]=O.map,z++,q.updateMatrices(O),O.castShadow&&D++),r.spotLightMatrix[E]=q.matrix,O.castShadow){const G=t.get(O);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,r.spotShadow[E]=G,r.spotShadowMap[E]=X,L++}E++}else if(O.isRectAreaLight){const Z=e.get(O);Z.color.copy(j).multiplyScalar(re),Z.halfWidth.set(O.width*.5,0,0),Z.halfHeight.set(0,O.height*.5,0),r.rectArea[S]=Z,S++}else if(O.isPointLight){const Z=e.get(O);if(Z.color.copy(O.color).multiplyScalar(O.intensity),Z.distance=O.distance,Z.decay=O.decay,O.castShadow){const q=O.shadow,G=t.get(O);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,G.shadowCameraNear=q.camera.near,G.shadowCameraFar=q.camera.far,r.pointShadow[y]=G,r.pointShadowMap[y]=X,r.pointShadowMatrix[y]=O.shadow.matrix,P++}r.point[y]=Z,y++}else if(O.isHemisphereLight){const Z=e.get(O);Z.skyColor.copy(O.color).multiplyScalar(re),Z.groundColor.copy(O.groundColor).multiplyScalar(re),r.hemi[v]=Z,v++}}S>0&&(s.has("OES_texture_float_linear")===!0?(r.rectAreaLTC1=Ue.LTC_FLOAT_1,r.rectAreaLTC2=Ue.LTC_FLOAT_2):(r.rectAreaLTC1=Ue.LTC_HALF_1,r.rectAreaLTC2=Ue.LTC_HALF_2)),r.ambient[0]=M,r.ambient[1]=u,r.ambient[2]=f;const R=r.hash;(R.directionalLength!==p||R.pointLength!==y||R.spotLength!==E||R.rectAreaLength!==S||R.hemiLength!==v||R.numDirectionalShadows!==A||R.numPointShadows!==P||R.numSpotShadows!==L||R.numSpotMaps!==z||R.numLightProbes!==F)&&(r.directional.length=p,r.spot.length=E,r.rectArea.length=S,r.point.length=y,r.hemi.length=v,r.directionalShadow.length=A,r.directionalShadowMap.length=A,r.pointShadow.length=P,r.pointShadowMap.length=P,r.spotShadow.length=L,r.spotShadowMap.length=L,r.directionalShadowMatrix.length=A,r.pointShadowMatrix.length=P,r.spotLightMatrix.length=L+z-D,r.spotLightMap.length=z,r.numSpotLightShadowsWithMaps=D,r.numLightProbes=F,R.directionalLength=p,R.pointLength=y,R.spotLength=E,R.rectAreaLength=S,R.hemiLength=v,R.numDirectionalShadows=A,R.numPointShadows=P,R.numSpotShadows=L,R.numSpotMaps=z,R.numLightProbes=F,r.version=kE++)}function g(_,M){let u=0,f=0,p=0,y=0,E=0;const S=M.matrixWorldInverse;for(let v=0,A=_.length;v<A;v++){const P=_[v];if(P.isDirectionalLight){const L=r.directional[u];L.direction.setFromMatrixPosition(P.matrixWorld),a.setFromMatrixPosition(P.target.matrixWorld),L.direction.sub(a),L.direction.transformDirection(S),u++}else if(P.isSpotLight){const L=r.spot[p];L.position.setFromMatrixPosition(P.matrixWorld),L.position.applyMatrix4(S),L.direction.setFromMatrixPosition(P.matrixWorld),a.setFromMatrixPosition(P.target.matrixWorld),L.direction.sub(a),L.direction.transformDirection(S),p++}else if(P.isRectAreaLight){const L=r.rectArea[y];L.position.setFromMatrixPosition(P.matrixWorld),L.position.applyMatrix4(S),d.identity(),l.copy(P.matrixWorld),l.premultiply(S),d.extractRotation(l),L.halfWidth.set(P.width*.5,0,0),L.halfHeight.set(0,P.height*.5,0),L.halfWidth.applyMatrix4(d),L.halfHeight.applyMatrix4(d),y++}else if(P.isPointLight){const L=r.point[f];L.position.setFromMatrixPosition(P.matrixWorld),L.position.applyMatrix4(S),f++}else if(P.isHemisphereLight){const L=r.hemi[E];L.direction.setFromMatrixPosition(P.matrixWorld),L.direction.transformDirection(S),E++}}}return{setup:m,setupView:g,state:r}}function n_(s){const e=new HE(s),t=[],r=[],a=[];function l(f){u.camera=f,t.length=0,r.length=0,a.length=0}function d(f){t.push(f)}function m(f){r.push(f)}function g(f){a.push(f)}function _(){e.setup(t)}function M(f){e.setupView(t,f)}const u={lightsArray:t,shadowsArray:r,lightProbeGridArray:a,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:l,state:u,setupLights:_,setupLightsView:M,pushLight:d,pushShadow:m,pushLightProbeGrid:g}}function VE(s){let e=new WeakMap;function t(a,l=0){const d=e.get(a);let m;return d===void 0?(m=new n_(s),e.set(a,[m])):l>=d.length?(m=new n_(s),d.push(m)):m=d[l],m}function r(){e=new WeakMap}return{get:t,dispose:r}}const GE=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,WE=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,XE=[new oe(1,0,0),new oe(-1,0,0),new oe(0,1,0),new oe(0,-1,0),new oe(0,0,1),new oe(0,0,-1)],YE=[new oe(0,-1,0),new oe(0,-1,0),new oe(0,0,1),new oe(0,0,-1),new oe(0,-1,0),new oe(0,-1,0)],i_=new rn,Ko=new oe,vf=new oe;function qE(s,e,t){let r=new L_;const a=new It,l=new It,d=new Jt,m=new ax,g=new lx,_={},M=t.maxTextureSize,u={[Dr]:kn,[kn]:Dr,[Zi]:Zi},f=new xi({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new It},radius:{value:4}},vertexShader:GE,fragmentShader:WE}),p=f.clone();p.defines.HORIZONTAL_PASS=1;const y=new ri;y.setAttribute("position",new Zt(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const E=new rr(y,f),S=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=kl;let v=this.type;this.render=function(D,F,R){if(S.enabled===!1||S.autoUpdate===!1&&S.needsUpdate===!1||D.length===0)return;this.type===U0&&(tt("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=kl);const I=s.getRenderTarget(),W=s.getActiveCubeFace(),O=s.getActiveMipmapLevel(),j=s.state;j.setBlending(Ji),j.buffers.depth.getReversed()===!0?j.buffers.color.setClear(0,0,0,0):j.buffers.color.setClear(1,1,1,1),j.buffers.depth.setTest(!0),j.setScissorTest(!1);const re=v!==this.type;re&&F.traverse(function(ae){ae.material&&(Array.isArray(ae.material)?ae.material.forEach(X=>X.needsUpdate=!0):ae.material.needsUpdate=!0)});for(let ae=0,X=D.length;ae<X;ae++){const Z=D[ae],q=Z.shadow;if(q===void 0){tt("WebGLShadowMap:",Z,"has no shadow.");continue}if(q.autoUpdate===!1&&q.needsUpdate===!1)continue;a.copy(q.mapSize);const G=q.getFrameExtents();a.multiply(G),l.copy(q.mapSize),(a.x>M||a.y>M)&&(a.x>M&&(l.x=Math.floor(M/G.x),a.x=l.x*G.x,q.mapSize.x=l.x),a.y>M&&(l.y=Math.floor(M/G.y),a.y=l.y*G.y,q.mapSize.y=l.y));const J=s.state.buffers.depth.getReversed();if(q.camera._reversedDepth=J,q.map===null||re===!0){if(q.map!==null&&(q.map.depthTexture!==null&&(q.map.depthTexture.dispose(),q.map.depthTexture=null),q.map.dispose()),this.type===Zo){if(Z.isPointLight){tt("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}q.map=new Ii(a.x,a.y,{format:os,type:nr,minFilter:Tn,magFilter:Tn,generateMipmaps:!1}),q.map.texture.name=Z.name+".shadowMap",q.map.depthTexture=new Js(a.x,a.y,Pi),q.map.depthTexture.name=Z.name+".shadowMapDepth",q.map.depthTexture.format=ir,q.map.depthTexture.compareFunction=null,q.map.depthTexture.minFilter=gn,q.map.depthTexture.magFilter=gn}else Z.isPointLight?(q.map=new z_(a.x),q.map.depthTexture=new tx(a.x,Ni)):(q.map=new Ii(a.x,a.y),q.map.depthTexture=new Js(a.x,a.y,Ni)),q.map.depthTexture.name=Z.name+".shadowMap",q.map.depthTexture.format=ir,this.type===kl?(q.map.depthTexture.compareFunction=J?Md:yd,q.map.depthTexture.minFilter=Tn,q.map.depthTexture.magFilter=Tn):(q.map.depthTexture.compareFunction=null,q.map.depthTexture.minFilter=gn,q.map.depthTexture.magFilter=gn);q.camera.updateProjectionMatrix()}const ie=q.map.isWebGLCubeRenderTarget?6:1;for(let U=0;U<ie;U++){if(q.map.isWebGLCubeRenderTarget)s.setRenderTarget(q.map,U),s.clear();else{U===0&&(s.setRenderTarget(q.map),s.clear());const K=q.getViewport(U);d.set(l.x*K.x,l.y*K.y,l.x*K.z,l.y*K.w),j.viewport(d)}if(Z.isPointLight){const K=q.camera,Le=q.matrix,De=Z.distance||K.far;De!==K.far&&(K.far=De,K.updateProjectionMatrix()),Ko.setFromMatrixPosition(Z.matrixWorld),K.position.copy(Ko),vf.copy(K.position),vf.add(XE[U]),K.up.copy(YE[U]),K.lookAt(vf),K.updateMatrixWorld(),Le.makeTranslation(-Ko.x,-Ko.y,-Ko.z),i_.multiplyMatrices(K.projectionMatrix,K.matrixWorldInverse),q._frustum.setFromProjectionMatrix(i_,K.coordinateSystem,K.reversedDepth)}else q.updateMatrices(Z);r=q.getFrustum(),L(F,R,q.camera,Z,this.type)}q.isPointLightShadow!==!0&&this.type===Zo&&A(q,R),q.needsUpdate=!1}v=this.type,S.needsUpdate=!1,s.setRenderTarget(I,W,O)};function A(D,F){const R=e.update(E);f.defines.VSM_SAMPLES!==D.blurSamples&&(f.defines.VSM_SAMPLES=D.blurSamples,p.defines.VSM_SAMPLES=D.blurSamples,f.needsUpdate=!0,p.needsUpdate=!0),D.mapPass===null&&(D.mapPass=new Ii(a.x,a.y,{format:os,type:nr})),f.uniforms.shadow_pass.value=D.map.depthTexture,f.uniforms.resolution.value=D.mapSize,f.uniforms.radius.value=D.radius,s.setRenderTarget(D.mapPass),s.clear(),s.renderBufferDirect(F,null,R,f,E,null),p.uniforms.shadow_pass.value=D.mapPass.texture,p.uniforms.resolution.value=D.mapSize,p.uniforms.radius.value=D.radius,s.setRenderTarget(D.map),s.clear(),s.renderBufferDirect(F,null,R,p,E,null)}function P(D,F,R,I){let W=null;const O=R.isPointLight===!0?D.customDistanceMaterial:D.customDepthMaterial;if(O!==void 0)W=O;else if(W=R.isPointLight===!0?g:m,s.localClippingEnabled&&F.clipShadows===!0&&Array.isArray(F.clippingPlanes)&&F.clippingPlanes.length!==0||F.displacementMap&&F.displacementScale!==0||F.alphaMap&&F.alphaTest>0||F.map&&F.alphaTest>0||F.alphaToCoverage===!0){const j=W.uuid,re=F.uuid;let ae=_[j];ae===void 0&&(ae={},_[j]=ae);let X=ae[re];X===void 0&&(X=W.clone(),ae[re]=X,F.addEventListener("dispose",z)),W=X}if(W.visible=F.visible,W.wireframe=F.wireframe,I===Zo?W.side=F.shadowSide!==null?F.shadowSide:F.side:W.side=F.shadowSide!==null?F.shadowSide:u[F.side],W.alphaMap=F.alphaMap,W.alphaTest=F.alphaToCoverage===!0?.5:F.alphaTest,W.map=F.map,W.clipShadows=F.clipShadows,W.clippingPlanes=F.clippingPlanes,W.clipIntersection=F.clipIntersection,W.displacementMap=F.displacementMap,W.displacementScale=F.displacementScale,W.displacementBias=F.displacementBias,W.wireframeLinewidth=F.wireframeLinewidth,W.linewidth=F.linewidth,R.isPointLight===!0&&W.isMeshDistanceMaterial===!0){const j=s.properties.get(W);j.light=R}return W}function L(D,F,R,I,W){if(D.visible===!1)return;if(D.layers.test(F.layers)&&(D.isMesh||D.isLine||D.isPoints)&&(D.castShadow||D.receiveShadow&&W===Zo)&&(!D.frustumCulled||r.intersectsObject(D))){D.modelViewMatrix.multiplyMatrices(R.matrixWorldInverse,D.matrixWorld);const re=e.update(D),ae=D.material;if(Array.isArray(ae)){const X=re.groups;for(let Z=0,q=X.length;Z<q;Z++){const G=X[Z],J=ae[G.materialIndex];if(J&&J.visible){const ie=P(D,J,I,W);D.onBeforeShadow(s,D,F,R,re,ie,G),s.renderBufferDirect(R,null,re,ie,D,G),D.onAfterShadow(s,D,F,R,re,ie,G)}}}else if(ae.visible){const X=P(D,ae,I,W);D.onBeforeShadow(s,D,F,R,re,X,null),s.renderBufferDirect(R,null,re,X,D,null),D.onAfterShadow(s,D,F,R,re,X,null)}}const j=D.children;for(let re=0,ae=j.length;re<ae;re++)L(j[re],F,R,I,W)}function z(D){D.target.removeEventListener("dispose",z);for(const R in _){const I=_[R],W=D.target.uuid;W in I&&(I[W].dispose(),delete I[W])}}}function jE(s,e){function t(){let k=!1;const Te=new Jt;let fe=null;const Oe=new Jt(0,0,0,0);return{setMask:function(Ce){fe!==Ce&&!k&&(s.colorMask(Ce,Ce,Ce,Ce),fe=Ce)},setLocked:function(Ce){k=Ce},setClear:function(Ce,ge,We,st,Nt){Nt===!0&&(Ce*=st,ge*=st,We*=st),Te.set(Ce,ge,We,st),Oe.equals(Te)===!1&&(s.clearColor(Ce,ge,We,st),Oe.copy(Te))},reset:function(){k=!1,fe=null,Oe.set(-1,0,0,0)}}}function r(){let k=!1,Te=!1,fe=null,Oe=null,Ce=null;return{setReversed:function(ge){if(Te!==ge){const We=e.get("EXT_clip_control");ge?We.clipControlEXT(We.LOWER_LEFT_EXT,We.ZERO_TO_ONE_EXT):We.clipControlEXT(We.LOWER_LEFT_EXT,We.NEGATIVE_ONE_TO_ONE_EXT),Te=ge;const st=Ce;Ce=null,this.setClear(st)}},getReversed:function(){return Te},setTest:function(ge){ge?de(s.DEPTH_TEST):Ie(s.DEPTH_TEST)},setMask:function(ge){fe!==ge&&!k&&(s.depthMask(ge),fe=ge)},setFunc:function(ge){if(Te&&(ge=pv[ge]),Oe!==ge){switch(ge){case Tf:s.depthFunc(s.NEVER);break;case wf:s.depthFunc(s.ALWAYS);break;case Af:s.depthFunc(s.LESS);break;case Zs:s.depthFunc(s.LEQUAL);break;case Rf:s.depthFunc(s.EQUAL);break;case Cf:s.depthFunc(s.GEQUAL);break;case bf:s.depthFunc(s.GREATER);break;case Pf:s.depthFunc(s.NOTEQUAL);break;default:s.depthFunc(s.LEQUAL)}Oe=ge}},setLocked:function(ge){k=ge},setClear:function(ge){Ce!==ge&&(Ce=ge,Te&&(ge=1-ge),s.clearDepth(ge))},reset:function(){k=!1,fe=null,Oe=null,Ce=null,Te=!1}}}function a(){let k=!1,Te=null,fe=null,Oe=null,Ce=null,ge=null,We=null,st=null,Nt=null;return{setTest:function(Tt){k||(Tt?de(s.STENCIL_TEST):Ie(s.STENCIL_TEST))},setMask:function(Tt){Te!==Tt&&!k&&(s.stencilMask(Tt),Te=Tt)},setFunc:function(Tt,wn,qn){(fe!==Tt||Oe!==wn||Ce!==qn)&&(s.stencilFunc(Tt,wn,qn),fe=Tt,Oe=wn,Ce=qn)},setOp:function(Tt,wn,qn){(ge!==Tt||We!==wn||st!==qn)&&(s.stencilOp(Tt,wn,qn),ge=Tt,We=wn,st=qn)},setLocked:function(Tt){k=Tt},setClear:function(Tt){Nt!==Tt&&(s.clearStencil(Tt),Nt=Tt)},reset:function(){k=!1,Te=null,fe=null,Oe=null,Ce=null,ge=null,We=null,st=null,Nt=null}}}const l=new t,d=new r,m=new a,g=new WeakMap,_=new WeakMap;let M={},u={},f={},p=new WeakMap,y=[],E=null,S=!1,v=null,A=null,P=null,L=null,z=null,D=null,F=null,R=new At(0,0,0),I=0,W=!1,O=null,j=null,re=null,ae=null,X=null;const Z=s.getParameter(s.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let q=!1,G=0;const J=s.getParameter(s.VERSION);J.indexOf("WebGL")!==-1?(G=parseFloat(/^WebGL (\d)/.exec(J)[1]),q=G>=1):J.indexOf("OpenGL ES")!==-1&&(G=parseFloat(/^OpenGL ES (\d)/.exec(J)[1]),q=G>=2);let ie=null,U={};const K=s.getParameter(s.SCISSOR_BOX),Le=s.getParameter(s.VIEWPORT),De=new Jt().fromArray(K),we=new Jt().fromArray(Le);function se(k,Te,fe,Oe){const Ce=new Uint8Array(4),ge=s.createTexture();s.bindTexture(k,ge),s.texParameteri(k,s.TEXTURE_MIN_FILTER,s.NEAREST),s.texParameteri(k,s.TEXTURE_MAG_FILTER,s.NEAREST);for(let We=0;We<fe;We++)k===s.TEXTURE_3D||k===s.TEXTURE_2D_ARRAY?s.texImage3D(Te,0,s.RGBA,1,1,Oe,0,s.RGBA,s.UNSIGNED_BYTE,Ce):s.texImage2D(Te+We,0,s.RGBA,1,1,0,s.RGBA,s.UNSIGNED_BYTE,Ce);return ge}const _e={};_e[s.TEXTURE_2D]=se(s.TEXTURE_2D,s.TEXTURE_2D,1),_e[s.TEXTURE_CUBE_MAP]=se(s.TEXTURE_CUBE_MAP,s.TEXTURE_CUBE_MAP_POSITIVE_X,6),_e[s.TEXTURE_2D_ARRAY]=se(s.TEXTURE_2D_ARRAY,s.TEXTURE_2D_ARRAY,1,1),_e[s.TEXTURE_3D]=se(s.TEXTURE_3D,s.TEXTURE_3D,1,1),l.setClear(0,0,0,1),d.setClear(1),m.setClear(0),de(s.DEPTH_TEST),d.setFunc(Zs),Yt(!1),Ft(im),de(s.CULL_FACE),Dt(Ji);function de(k){M[k]!==!0&&(s.enable(k),M[k]=!0)}function Ie(k){M[k]!==!1&&(s.disable(k),M[k]=!1)}function je(k,Te){return f[k]!==Te?(s.bindFramebuffer(k,Te),f[k]=Te,k===s.DRAW_FRAMEBUFFER&&(f[s.FRAMEBUFFER]=Te),k===s.FRAMEBUFFER&&(f[s.DRAW_FRAMEBUFFER]=Te),!0):!1}function $e(k,Te){let fe=y,Oe=!1;if(k){fe=p.get(Te),fe===void 0&&(fe=[],p.set(Te,fe));const Ce=k.textures;if(fe.length!==Ce.length||fe[0]!==s.COLOR_ATTACHMENT0){for(let ge=0,We=Ce.length;ge<We;ge++)fe[ge]=s.COLOR_ATTACHMENT0+ge;fe.length=Ce.length,Oe=!0}}else fe[0]!==s.BACK&&(fe[0]=s.BACK,Oe=!0);Oe&&s.drawBuffers(fe)}function Ut(k){return E!==k?(s.useProgram(k),E=k,!0):!1}const ct={[es]:s.FUNC_ADD,[O0]:s.FUNC_SUBTRACT,[B0]:s.FUNC_REVERSE_SUBTRACT};ct[k0]=s.MIN,ct[z0]=s.MAX;const Et={[H0]:s.ZERO,[V0]:s.ONE,[G0]:s.SRC_COLOR,[Mf]:s.SRC_ALPHA,[K0]:s.SRC_ALPHA_SATURATE,[q0]:s.DST_COLOR,[X0]:s.DST_ALPHA,[W0]:s.ONE_MINUS_SRC_COLOR,[Ef]:s.ONE_MINUS_SRC_ALPHA,[j0]:s.ONE_MINUS_DST_COLOR,[Y0]:s.ONE_MINUS_DST_ALPHA,[$0]:s.CONSTANT_COLOR,[Z0]:s.ONE_MINUS_CONSTANT_COLOR,[Q0]:s.CONSTANT_ALPHA,[J0]:s.ONE_MINUS_CONSTANT_ALPHA};function Dt(k,Te,fe,Oe,Ce,ge,We,st,Nt,Tt){if(k===Ji){S===!0&&(Ie(s.BLEND),S=!1);return}if(S===!1&&(de(s.BLEND),S=!0),k!==F0){if(k!==v||Tt!==W){if((A!==es||z!==es)&&(s.blendEquation(s.FUNC_ADD),A=es,z=es),Tt)switch(k){case Ks:s.blendFuncSeparate(s.ONE,s.ONE_MINUS_SRC_ALPHA,s.ONE,s.ONE_MINUS_SRC_ALPHA);break;case Xl:s.blendFunc(s.ONE,s.ONE);break;case rm:s.blendFuncSeparate(s.ZERO,s.ONE_MINUS_SRC_COLOR,s.ZERO,s.ONE);break;case sm:s.blendFuncSeparate(s.DST_COLOR,s.ONE_MINUS_SRC_ALPHA,s.ZERO,s.ONE);break;default:Mt("WebGLState: Invalid blending: ",k);break}else switch(k){case Ks:s.blendFuncSeparate(s.SRC_ALPHA,s.ONE_MINUS_SRC_ALPHA,s.ONE,s.ONE_MINUS_SRC_ALPHA);break;case Xl:s.blendFuncSeparate(s.SRC_ALPHA,s.ONE,s.ONE,s.ONE);break;case rm:Mt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case sm:Mt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Mt("WebGLState: Invalid blending: ",k);break}P=null,L=null,D=null,F=null,R.set(0,0,0),I=0,v=k,W=Tt}return}Ce=Ce||Te,ge=ge||fe,We=We||Oe,(Te!==A||Ce!==z)&&(s.blendEquationSeparate(ct[Te],ct[Ce]),A=Te,z=Ce),(fe!==P||Oe!==L||ge!==D||We!==F)&&(s.blendFuncSeparate(Et[fe],Et[Oe],Et[ge],Et[We]),P=fe,L=Oe,D=ge,F=We),(st.equals(R)===!1||Nt!==I)&&(s.blendColor(st.r,st.g,st.b,Nt),R.copy(st),I=Nt),v=k,W=!1}function ft(k,Te){k.side===Zi?Ie(s.CULL_FACE):de(s.CULL_FACE);let fe=k.side===kn;Te&&(fe=!fe),Yt(fe),k.blending===Ks&&k.transparent===!1?Dt(Ji):Dt(k.blending,k.blendEquation,k.blendSrc,k.blendDst,k.blendEquationAlpha,k.blendSrcAlpha,k.blendDstAlpha,k.blendColor,k.blendAlpha,k.premultipliedAlpha),d.setFunc(k.depthFunc),d.setTest(k.depthTest),d.setMask(k.depthWrite),l.setMask(k.colorWrite);const Oe=k.stencilWrite;m.setTest(Oe),Oe&&(m.setMask(k.stencilWriteMask),m.setFunc(k.stencilFunc,k.stencilRef,k.stencilFuncMask),m.setOp(k.stencilFail,k.stencilZFail,k.stencilZPass)),H(k.polygonOffset,k.polygonOffsetFactor,k.polygonOffsetUnits),k.alphaToCoverage===!0?de(s.SAMPLE_ALPHA_TO_COVERAGE):Ie(s.SAMPLE_ALPHA_TO_COVERAGE)}function Yt(k){O!==k&&(k?s.frontFace(s.CW):s.frontFace(s.CCW),O=k)}function Ft(k){k!==I0?(de(s.CULL_FACE),k!==j&&(k===im?s.cullFace(s.BACK):k===N0?s.cullFace(s.FRONT):s.cullFace(s.FRONT_AND_BACK))):Ie(s.CULL_FACE),j=k}function hn(k){k!==re&&(q&&s.lineWidth(k),re=k)}function H(k,Te,fe){k?(de(s.POLYGON_OFFSET_FILL),(ae!==Te||X!==fe)&&(ae=Te,X=fe,d.getReversed()&&(Te=-Te),s.polygonOffset(Te,fe))):Ie(s.POLYGON_OFFSET_FILL)}function Ot(k){k?de(s.SCISSOR_TEST):Ie(s.SCISSOR_TEST)}function dt(k){k===void 0&&(k=s.TEXTURE0+Z-1),ie!==k&&(s.activeTexture(k),ie=k)}function Ct(k,Te,fe){fe===void 0&&(ie===null?fe=s.TEXTURE0+Z-1:fe=ie);let Oe=U[fe];Oe===void 0&&(Oe={type:void 0,texture:void 0},U[fe]=Oe),(Oe.type!==k||Oe.texture!==Te)&&(ie!==fe&&(s.activeTexture(fe),ie=fe),s.bindTexture(k,Te||_e[k]),Oe.type=k,Oe.texture=Te)}function Ne(){const k=U[ie];k!==void 0&&k.type!==void 0&&(s.bindTexture(k.type,null),k.type=void 0,k.texture=void 0)}function zt(){try{s.compressedTexImage2D(...arguments)}catch(k){Mt("WebGLState:",k)}}function b(){try{s.compressedTexImage3D(...arguments)}catch(k){Mt("WebGLState:",k)}}function T(){try{s.texSubImage2D(...arguments)}catch(k){Mt("WebGLState:",k)}}function $(){try{s.texSubImage3D(...arguments)}catch(k){Mt("WebGLState:",k)}}function he(){try{s.compressedTexSubImage2D(...arguments)}catch(k){Mt("WebGLState:",k)}}function me(){try{s.compressedTexSubImage3D(...arguments)}catch(k){Mt("WebGLState:",k)}}function ye(){try{s.texStorage2D(...arguments)}catch(k){Mt("WebGLState:",k)}}function Pe(){try{s.texStorage3D(...arguments)}catch(k){Mt("WebGLState:",k)}}function ce(){try{s.texImage2D(...arguments)}catch(k){Mt("WebGLState:",k)}}function pe(){try{s.texImage3D(...arguments)}catch(k){Mt("WebGLState:",k)}}function Fe(k){return u[k]!==void 0?u[k]:s.getParameter(k)}function Be(k,Te){u[k]!==Te&&(s.pixelStorei(k,Te),u[k]=Te)}function Ae(k){De.equals(k)===!1&&(s.scissor(k.x,k.y,k.z,k.w),De.copy(k))}function Me(k){we.equals(k)===!1&&(s.viewport(k.x,k.y,k.z,k.w),we.copy(k))}function et(k,Te){let fe=_.get(Te);fe===void 0&&(fe=new WeakMap,_.set(Te,fe));let Oe=fe.get(k);Oe===void 0&&(Oe=s.getUniformBlockIndex(Te,k.name),fe.set(k,Oe))}function rt(k,Te){const Oe=_.get(Te).get(k);g.get(Te)!==Oe&&(s.uniformBlockBinding(Te,Oe,k.__bindingPointIndex),g.set(Te,Oe))}function pt(){s.disable(s.BLEND),s.disable(s.CULL_FACE),s.disable(s.DEPTH_TEST),s.disable(s.POLYGON_OFFSET_FILL),s.disable(s.SCISSOR_TEST),s.disable(s.STENCIL_TEST),s.disable(s.SAMPLE_ALPHA_TO_COVERAGE),s.blendEquation(s.FUNC_ADD),s.blendFunc(s.ONE,s.ZERO),s.blendFuncSeparate(s.ONE,s.ZERO,s.ONE,s.ZERO),s.blendColor(0,0,0,0),s.colorMask(!0,!0,!0,!0),s.clearColor(0,0,0,0),s.depthMask(!0),s.depthFunc(s.LESS),d.setReversed(!1),s.clearDepth(1),s.stencilMask(4294967295),s.stencilFunc(s.ALWAYS,0,4294967295),s.stencilOp(s.KEEP,s.KEEP,s.KEEP),s.clearStencil(0),s.cullFace(s.BACK),s.frontFace(s.CCW),s.polygonOffset(0,0),s.activeTexture(s.TEXTURE0),s.bindFramebuffer(s.FRAMEBUFFER,null),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,null),s.bindFramebuffer(s.READ_FRAMEBUFFER,null),s.useProgram(null),s.lineWidth(1),s.scissor(0,0,s.canvas.width,s.canvas.height),s.viewport(0,0,s.canvas.width,s.canvas.height),s.pixelStorei(s.PACK_ALIGNMENT,4),s.pixelStorei(s.UNPACK_ALIGNMENT,4),s.pixelStorei(s.UNPACK_FLIP_Y_WEBGL,!1),s.pixelStorei(s.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),s.pixelStorei(s.UNPACK_COLORSPACE_CONVERSION_WEBGL,s.BROWSER_DEFAULT_WEBGL),s.pixelStorei(s.PACK_ROW_LENGTH,0),s.pixelStorei(s.PACK_SKIP_PIXELS,0),s.pixelStorei(s.PACK_SKIP_ROWS,0),s.pixelStorei(s.UNPACK_ROW_LENGTH,0),s.pixelStorei(s.UNPACK_IMAGE_HEIGHT,0),s.pixelStorei(s.UNPACK_SKIP_PIXELS,0),s.pixelStorei(s.UNPACK_SKIP_ROWS,0),s.pixelStorei(s.UNPACK_SKIP_IMAGES,0),M={},u={},ie=null,U={},f={},p=new WeakMap,y=[],E=null,S=!1,v=null,A=null,P=null,L=null,z=null,D=null,F=null,R=new At(0,0,0),I=0,W=!1,O=null,j=null,re=null,ae=null,X=null,De.set(0,0,s.canvas.width,s.canvas.height),we.set(0,0,s.canvas.width,s.canvas.height),l.reset(),d.reset(),m.reset()}return{buffers:{color:l,depth:d,stencil:m},enable:de,disable:Ie,bindFramebuffer:je,drawBuffers:$e,useProgram:Ut,setBlending:Dt,setMaterial:ft,setFlipSided:Yt,setCullFace:Ft,setLineWidth:hn,setPolygonOffset:H,setScissorTest:Ot,activeTexture:dt,bindTexture:Ct,unbindTexture:Ne,compressedTexImage2D:zt,compressedTexImage3D:b,texImage2D:ce,texImage3D:pe,pixelStorei:Be,getParameter:Fe,updateUBOMapping:et,uniformBlockBinding:rt,texStorage2D:ye,texStorage3D:Pe,texSubImage2D:T,texSubImage3D:$,compressedTexSubImage2D:he,compressedTexSubImage3D:me,scissor:Ae,viewport:Me,reset:pt}}function KE(s,e,t,r,a,l,d){const m=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,g=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),_=new It,M=new WeakMap,u=new Set;let f;const p=new WeakMap;let y=!1;try{y=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function E(b,T){return y?new OffscreenCanvas(b,T):Zl("canvas")}function S(b,T,$){let he=1;const me=zt(b);if((me.width>$||me.height>$)&&(he=$/Math.max(me.width,me.height)),he<1)if(typeof HTMLImageElement<"u"&&b instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&b instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&b instanceof ImageBitmap||typeof VideoFrame<"u"&&b instanceof VideoFrame){const ye=Math.floor(he*me.width),Pe=Math.floor(he*me.height);f===void 0&&(f=E(ye,Pe));const ce=T?E(ye,Pe):f;return ce.width=ye,ce.height=Pe,ce.getContext("2d").drawImage(b,0,0,ye,Pe),tt("WebGLRenderer: Texture has been resized from ("+me.width+"x"+me.height+") to ("+ye+"x"+Pe+")."),ce}else return"data"in b&&tt("WebGLRenderer: Image in DataTexture is too big ("+me.width+"x"+me.height+")."),b;return b}function v(b){return b.generateMipmaps}function A(b){s.generateMipmap(b)}function P(b){return b.isWebGLCubeRenderTarget?s.TEXTURE_CUBE_MAP:b.isWebGL3DRenderTarget?s.TEXTURE_3D:b.isWebGLArrayRenderTarget||b.isCompressedArrayTexture?s.TEXTURE_2D_ARRAY:s.TEXTURE_2D}function L(b,T,$,he,me,ye=!1){if(b!==null){if(s[b]!==void 0)return s[b];tt("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+b+"'")}let Pe;he&&(Pe=e.get("EXT_texture_norm16"),Pe||tt("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let ce=T;if(T===s.RED&&($===s.FLOAT&&(ce=s.R32F),$===s.HALF_FLOAT&&(ce=s.R16F),$===s.UNSIGNED_BYTE&&(ce=s.R8),$===s.UNSIGNED_SHORT&&Pe&&(ce=Pe.R16_EXT),$===s.SHORT&&Pe&&(ce=Pe.R16_SNORM_EXT)),T===s.RED_INTEGER&&($===s.UNSIGNED_BYTE&&(ce=s.R8UI),$===s.UNSIGNED_SHORT&&(ce=s.R16UI),$===s.UNSIGNED_INT&&(ce=s.R32UI),$===s.BYTE&&(ce=s.R8I),$===s.SHORT&&(ce=s.R16I),$===s.INT&&(ce=s.R32I)),T===s.RG&&($===s.FLOAT&&(ce=s.RG32F),$===s.HALF_FLOAT&&(ce=s.RG16F),$===s.UNSIGNED_BYTE&&(ce=s.RG8),$===s.UNSIGNED_SHORT&&Pe&&(ce=Pe.RG16_EXT),$===s.SHORT&&Pe&&(ce=Pe.RG16_SNORM_EXT)),T===s.RG_INTEGER&&($===s.UNSIGNED_BYTE&&(ce=s.RG8UI),$===s.UNSIGNED_SHORT&&(ce=s.RG16UI),$===s.UNSIGNED_INT&&(ce=s.RG32UI),$===s.BYTE&&(ce=s.RG8I),$===s.SHORT&&(ce=s.RG16I),$===s.INT&&(ce=s.RG32I)),T===s.RGB_INTEGER&&($===s.UNSIGNED_BYTE&&(ce=s.RGB8UI),$===s.UNSIGNED_SHORT&&(ce=s.RGB16UI),$===s.UNSIGNED_INT&&(ce=s.RGB32UI),$===s.BYTE&&(ce=s.RGB8I),$===s.SHORT&&(ce=s.RGB16I),$===s.INT&&(ce=s.RGB32I)),T===s.RGBA_INTEGER&&($===s.UNSIGNED_BYTE&&(ce=s.RGBA8UI),$===s.UNSIGNED_SHORT&&(ce=s.RGBA16UI),$===s.UNSIGNED_INT&&(ce=s.RGBA32UI),$===s.BYTE&&(ce=s.RGBA8I),$===s.SHORT&&(ce=s.RGBA16I),$===s.INT&&(ce=s.RGBA32I)),T===s.RGB&&($===s.UNSIGNED_SHORT&&Pe&&(ce=Pe.RGB16_EXT),$===s.SHORT&&Pe&&(ce=Pe.RGB16_SNORM_EXT),$===s.UNSIGNED_INT_5_9_9_9_REV&&(ce=s.RGB9_E5),$===s.UNSIGNED_INT_10F_11F_11F_REV&&(ce=s.R11F_G11F_B10F)),T===s.RGBA){const pe=ye?Kl:xt.getTransfer(me);$===s.FLOAT&&(ce=s.RGBA32F),$===s.HALF_FLOAT&&(ce=s.RGBA16F),$===s.UNSIGNED_BYTE&&(ce=pe===Lt?s.SRGB8_ALPHA8:s.RGBA8),$===s.UNSIGNED_SHORT&&Pe&&(ce=Pe.RGBA16_EXT),$===s.SHORT&&Pe&&(ce=Pe.RGBA16_SNORM_EXT),$===s.UNSIGNED_SHORT_4_4_4_4&&(ce=s.RGBA4),$===s.UNSIGNED_SHORT_5_5_5_1&&(ce=s.RGB5_A1)}return(ce===s.R16F||ce===s.R32F||ce===s.RG16F||ce===s.RG32F||ce===s.RGBA16F||ce===s.RGBA32F)&&e.get("EXT_color_buffer_float"),ce}function z(b,T){let $;return b?T===null||T===Ni||T===ia?$=s.DEPTH24_STENCIL8:T===Pi?$=s.DEPTH32F_STENCIL8:T===na&&($=s.DEPTH24_STENCIL8,tt("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===Ni||T===ia?$=s.DEPTH_COMPONENT24:T===Pi?$=s.DEPTH_COMPONENT32F:T===na&&($=s.DEPTH_COMPONENT16),$}function D(b,T){return v(b)===!0||b.isFramebufferTexture&&b.minFilter!==gn&&b.minFilter!==Tn?Math.log2(Math.max(T.width,T.height))+1:b.mipmaps!==void 0&&b.mipmaps.length>0?b.mipmaps.length:b.isCompressedTexture&&Array.isArray(b.image)?T.mipmaps.length:1}function F(b){const T=b.target;T.removeEventListener("dispose",F),I(T),T.isVideoTexture&&M.delete(T),T.isHTMLTexture&&u.delete(T)}function R(b){const T=b.target;T.removeEventListener("dispose",R),O(T)}function I(b){const T=r.get(b);if(T.__webglInit===void 0)return;const $=b.source,he=p.get($);if(he){const me=he[T.__cacheKey];me.usedTimes--,me.usedTimes===0&&W(b),Object.keys(he).length===0&&p.delete($)}r.remove(b)}function W(b){const T=r.get(b);s.deleteTexture(T.__webglTexture);const $=b.source,he=p.get($);delete he[T.__cacheKey],d.memory.textures--}function O(b){const T=r.get(b);if(b.depthTexture&&(b.depthTexture.dispose(),r.remove(b.depthTexture)),b.isWebGLCubeRenderTarget)for(let he=0;he<6;he++){if(Array.isArray(T.__webglFramebuffer[he]))for(let me=0;me<T.__webglFramebuffer[he].length;me++)s.deleteFramebuffer(T.__webglFramebuffer[he][me]);else s.deleteFramebuffer(T.__webglFramebuffer[he]);T.__webglDepthbuffer&&s.deleteRenderbuffer(T.__webglDepthbuffer[he])}else{if(Array.isArray(T.__webglFramebuffer))for(let he=0;he<T.__webglFramebuffer.length;he++)s.deleteFramebuffer(T.__webglFramebuffer[he]);else s.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&s.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&s.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let he=0;he<T.__webglColorRenderbuffer.length;he++)T.__webglColorRenderbuffer[he]&&s.deleteRenderbuffer(T.__webglColorRenderbuffer[he]);T.__webglDepthRenderbuffer&&s.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const $=b.textures;for(let he=0,me=$.length;he<me;he++){const ye=r.get($[he]);ye.__webglTexture&&(s.deleteTexture(ye.__webglTexture),d.memory.textures--),r.remove($[he])}r.remove(b)}let j=0;function re(){j=0}function ae(){return j}function X(b){j=b}function Z(){const b=j;return b>=a.maxTextures&&tt("WebGLTextures: Trying to use "+b+" texture units while this GPU supports only "+a.maxTextures),j+=1,b}function q(b){const T=[];return T.push(b.wrapS),T.push(b.wrapT),T.push(b.wrapR||0),T.push(b.magFilter),T.push(b.minFilter),T.push(b.anisotropy),T.push(b.internalFormat),T.push(b.format),T.push(b.type),T.push(b.generateMipmaps),T.push(b.premultiplyAlpha),T.push(b.flipY),T.push(b.unpackAlignment),T.push(b.colorSpace),T.join()}function G(b,T){const $=r.get(b);if(b.isVideoTexture&&Ct(b),b.isRenderTargetTexture===!1&&b.isExternalTexture!==!0&&b.version>0&&$.__version!==b.version){const he=b.image;if(he===null)tt("WebGLRenderer: Texture marked for update but no image data found.");else if(he.complete===!1)tt("WebGLRenderer: Texture marked for update but image is incomplete");else{Ie($,b,T);return}}else b.isExternalTexture&&($.__webglTexture=b.sourceTexture?b.sourceTexture:null);t.bindTexture(s.TEXTURE_2D,$.__webglTexture,s.TEXTURE0+T)}function J(b,T){const $=r.get(b);if(b.isRenderTargetTexture===!1&&b.version>0&&$.__version!==b.version){Ie($,b,T);return}else b.isExternalTexture&&($.__webglTexture=b.sourceTexture?b.sourceTexture:null);t.bindTexture(s.TEXTURE_2D_ARRAY,$.__webglTexture,s.TEXTURE0+T)}function ie(b,T){const $=r.get(b);if(b.isRenderTargetTexture===!1&&b.version>0&&$.__version!==b.version){Ie($,b,T);return}t.bindTexture(s.TEXTURE_3D,$.__webglTexture,s.TEXTURE0+T)}function U(b,T){const $=r.get(b);if(b.isCubeDepthTexture!==!0&&b.version>0&&$.__version!==b.version){je($,b,T);return}t.bindTexture(s.TEXTURE_CUBE_MAP,$.__webglTexture,s.TEXTURE0+T)}const K={[Lf]:s.REPEAT,[Qi]:s.CLAMP_TO_EDGE,[Df]:s.MIRRORED_REPEAT},Le={[gn]:s.NEAREST,[nv]:s.NEAREST_MIPMAP_NEAREST,[dl]:s.NEAREST_MIPMAP_LINEAR,[Tn]:s.LINEAR,[Gc]:s.LINEAR_MIPMAP_NEAREST,[ns]:s.LINEAR_MIPMAP_LINEAR},De={[sv]:s.NEVER,[cv]:s.ALWAYS,[ov]:s.LESS,[yd]:s.LEQUAL,[av]:s.EQUAL,[Md]:s.GEQUAL,[lv]:s.GREATER,[uv]:s.NOTEQUAL};function we(b,T){if(T.type===Pi&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===Tn||T.magFilter===Gc||T.magFilter===dl||T.magFilter===ns||T.minFilter===Tn||T.minFilter===Gc||T.minFilter===dl||T.minFilter===ns)&&tt("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),s.texParameteri(b,s.TEXTURE_WRAP_S,K[T.wrapS]),s.texParameteri(b,s.TEXTURE_WRAP_T,K[T.wrapT]),(b===s.TEXTURE_3D||b===s.TEXTURE_2D_ARRAY)&&s.texParameteri(b,s.TEXTURE_WRAP_R,K[T.wrapR]),s.texParameteri(b,s.TEXTURE_MAG_FILTER,Le[T.magFilter]),s.texParameteri(b,s.TEXTURE_MIN_FILTER,Le[T.minFilter]),T.compareFunction&&(s.texParameteri(b,s.TEXTURE_COMPARE_MODE,s.COMPARE_REF_TO_TEXTURE),s.texParameteri(b,s.TEXTURE_COMPARE_FUNC,De[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===gn||T.minFilter!==dl&&T.minFilter!==ns||T.type===Pi&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||r.get(T).__currentAnisotropy){const $=e.get("EXT_texture_filter_anisotropic");s.texParameterf(b,$.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,a.getMaxAnisotropy())),r.get(T).__currentAnisotropy=T.anisotropy}}}function se(b,T){let $=!1;b.__webglInit===void 0&&(b.__webglInit=!0,T.addEventListener("dispose",F));const he=T.source;let me=p.get(he);me===void 0&&(me={},p.set(he,me));const ye=q(T);if(ye!==b.__cacheKey){me[ye]===void 0&&(me[ye]={texture:s.createTexture(),usedTimes:0},d.memory.textures++,$=!0),me[ye].usedTimes++;const Pe=me[b.__cacheKey];Pe!==void 0&&(me[b.__cacheKey].usedTimes--,Pe.usedTimes===0&&W(T)),b.__cacheKey=ye,b.__webglTexture=me[ye].texture}return $}function _e(b,T,$){return Math.floor(Math.floor(b/$)/T)}function de(b,T,$,he){const ye=b.updateRanges;if(ye.length===0)t.texSubImage2D(s.TEXTURE_2D,0,0,0,T.width,T.height,$,he,T.data);else{ye.sort((Be,Ae)=>Be.start-Ae.start);let Pe=0;for(let Be=1;Be<ye.length;Be++){const Ae=ye[Pe],Me=ye[Be],et=Ae.start+Ae.count,rt=_e(Me.start,T.width,4),pt=_e(Ae.start,T.width,4);Me.start<=et+1&&rt===pt&&_e(Me.start+Me.count-1,T.width,4)===rt?Ae.count=Math.max(Ae.count,Me.start+Me.count-Ae.start):(++Pe,ye[Pe]=Me)}ye.length=Pe+1;const ce=t.getParameter(s.UNPACK_ROW_LENGTH),pe=t.getParameter(s.UNPACK_SKIP_PIXELS),Fe=t.getParameter(s.UNPACK_SKIP_ROWS);t.pixelStorei(s.UNPACK_ROW_LENGTH,T.width);for(let Be=0,Ae=ye.length;Be<Ae;Be++){const Me=ye[Be],et=Math.floor(Me.start/4),rt=Math.ceil(Me.count/4),pt=et%T.width,k=Math.floor(et/T.width),Te=rt,fe=1;t.pixelStorei(s.UNPACK_SKIP_PIXELS,pt),t.pixelStorei(s.UNPACK_SKIP_ROWS,k),t.texSubImage2D(s.TEXTURE_2D,0,pt,k,Te,fe,$,he,T.data)}b.clearUpdateRanges(),t.pixelStorei(s.UNPACK_ROW_LENGTH,ce),t.pixelStorei(s.UNPACK_SKIP_PIXELS,pe),t.pixelStorei(s.UNPACK_SKIP_ROWS,Fe)}}function Ie(b,T,$){let he=s.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(he=s.TEXTURE_2D_ARRAY),T.isData3DTexture&&(he=s.TEXTURE_3D);const me=se(b,T),ye=T.source;t.bindTexture(he,b.__webglTexture,s.TEXTURE0+$);const Pe=r.get(ye);if(ye.version!==Pe.__version||me===!0){if(t.activeTexture(s.TEXTURE0+$),(typeof ImageBitmap<"u"&&T.image instanceof ImageBitmap)===!1){const fe=xt.getPrimaries(xt.workingColorSpace),Oe=T.colorSpace===Pr?null:xt.getPrimaries(T.colorSpace),Ce=T.colorSpace===Pr||fe===Oe?s.NONE:s.BROWSER_DEFAULT_WEBGL;t.pixelStorei(s.UNPACK_FLIP_Y_WEBGL,T.flipY),t.pixelStorei(s.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),t.pixelStorei(s.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ce)}t.pixelStorei(s.UNPACK_ALIGNMENT,T.unpackAlignment);let pe=S(T.image,!1,a.maxTextureSize);pe=Ne(T,pe);const Fe=l.convert(T.format,T.colorSpace),Be=l.convert(T.type);let Ae=L(T.internalFormat,Fe,Be,T.normalized,T.colorSpace,T.isVideoTexture);we(he,T);let Me;const et=T.mipmaps,rt=T.isVideoTexture!==!0,pt=Pe.__version===void 0||me===!0,k=ye.dataReady,Te=D(T,pe);if(T.isDepthTexture)Ae=z(T.format===is,T.type),pt&&(rt?t.texStorage2D(s.TEXTURE_2D,1,Ae,pe.width,pe.height):t.texImage2D(s.TEXTURE_2D,0,Ae,pe.width,pe.height,0,Fe,Be,null));else if(T.isDataTexture)if(et.length>0){rt&&pt&&t.texStorage2D(s.TEXTURE_2D,Te,Ae,et[0].width,et[0].height);for(let fe=0,Oe=et.length;fe<Oe;fe++)Me=et[fe],rt?k&&t.texSubImage2D(s.TEXTURE_2D,fe,0,0,Me.width,Me.height,Fe,Be,Me.data):t.texImage2D(s.TEXTURE_2D,fe,Ae,Me.width,Me.height,0,Fe,Be,Me.data);T.generateMipmaps=!1}else rt?(pt&&t.texStorage2D(s.TEXTURE_2D,Te,Ae,pe.width,pe.height),k&&de(T,pe,Fe,Be)):t.texImage2D(s.TEXTURE_2D,0,Ae,pe.width,pe.height,0,Fe,Be,pe.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){rt&&pt&&t.texStorage3D(s.TEXTURE_2D_ARRAY,Te,Ae,et[0].width,et[0].height,pe.depth);for(let fe=0,Oe=et.length;fe<Oe;fe++)if(Me=et[fe],T.format!==gi)if(Fe!==null)if(rt){if(k)if(T.layerUpdates.size>0){const Ce=Nm(Me.width,Me.height,T.format,T.type);for(const ge of T.layerUpdates){const We=Me.data.subarray(ge*Ce/Me.data.BYTES_PER_ELEMENT,(ge+1)*Ce/Me.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(s.TEXTURE_2D_ARRAY,fe,0,0,ge,Me.width,Me.height,1,Fe,We)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(s.TEXTURE_2D_ARRAY,fe,0,0,0,Me.width,Me.height,pe.depth,Fe,Me.data)}else t.compressedTexImage3D(s.TEXTURE_2D_ARRAY,fe,Ae,Me.width,Me.height,pe.depth,0,Me.data,0,0);else tt("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else rt?k&&t.texSubImage3D(s.TEXTURE_2D_ARRAY,fe,0,0,0,Me.width,Me.height,pe.depth,Fe,Be,Me.data):t.texImage3D(s.TEXTURE_2D_ARRAY,fe,Ae,Me.width,Me.height,pe.depth,0,Fe,Be,Me.data)}else{rt&&pt&&t.texStorage2D(s.TEXTURE_2D,Te,Ae,et[0].width,et[0].height);for(let fe=0,Oe=et.length;fe<Oe;fe++)Me=et[fe],T.format!==gi?Fe!==null?rt?k&&t.compressedTexSubImage2D(s.TEXTURE_2D,fe,0,0,Me.width,Me.height,Fe,Me.data):t.compressedTexImage2D(s.TEXTURE_2D,fe,Ae,Me.width,Me.height,0,Me.data):tt("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):rt?k&&t.texSubImage2D(s.TEXTURE_2D,fe,0,0,Me.width,Me.height,Fe,Be,Me.data):t.texImage2D(s.TEXTURE_2D,fe,Ae,Me.width,Me.height,0,Fe,Be,Me.data)}else if(T.isDataArrayTexture)if(rt){if(pt&&t.texStorage3D(s.TEXTURE_2D_ARRAY,Te,Ae,pe.width,pe.height,pe.depth),k)if(T.layerUpdates.size>0){const fe=Nm(pe.width,pe.height,T.format,T.type);for(const Oe of T.layerUpdates){const Ce=pe.data.subarray(Oe*fe/pe.data.BYTES_PER_ELEMENT,(Oe+1)*fe/pe.data.BYTES_PER_ELEMENT);t.texSubImage3D(s.TEXTURE_2D_ARRAY,0,0,0,Oe,pe.width,pe.height,1,Fe,Be,Ce)}T.clearLayerUpdates()}else t.texSubImage3D(s.TEXTURE_2D_ARRAY,0,0,0,0,pe.width,pe.height,pe.depth,Fe,Be,pe.data)}else t.texImage3D(s.TEXTURE_2D_ARRAY,0,Ae,pe.width,pe.height,pe.depth,0,Fe,Be,pe.data);else if(T.isData3DTexture)rt?(pt&&t.texStorage3D(s.TEXTURE_3D,Te,Ae,pe.width,pe.height,pe.depth),k&&t.texSubImage3D(s.TEXTURE_3D,0,0,0,0,pe.width,pe.height,pe.depth,Fe,Be,pe.data)):t.texImage3D(s.TEXTURE_3D,0,Ae,pe.width,pe.height,pe.depth,0,Fe,Be,pe.data);else if(T.isFramebufferTexture){if(pt)if(rt)t.texStorage2D(s.TEXTURE_2D,Te,Ae,pe.width,pe.height);else{let fe=pe.width,Oe=pe.height;for(let Ce=0;Ce<Te;Ce++)t.texImage2D(s.TEXTURE_2D,Ce,Ae,fe,Oe,0,Fe,Be,null),fe>>=1,Oe>>=1}}else if(T.isHTMLTexture){if("texElementImage2D"in s){const fe=s.canvas;if(fe.hasAttribute("layoutsubtree")||fe.setAttribute("layoutsubtree","true"),pe.parentNode!==fe){fe.appendChild(pe),u.add(T),fe.onpaint=st=>{const Nt=st.changedElements;for(const Tt of u)Nt.includes(Tt.image)&&(Tt.needsUpdate=!0)},fe.requestPaint();return}const Oe=0,Ce=s.RGBA,ge=s.RGBA,We=s.UNSIGNED_BYTE;s.texElementImage2D(s.TEXTURE_2D,Oe,Ce,ge,We,pe),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MIN_FILTER,s.LINEAR),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_S,s.CLAMP_TO_EDGE),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_T,s.CLAMP_TO_EDGE)}}else if(et.length>0){if(rt&&pt){const fe=zt(et[0]);t.texStorage2D(s.TEXTURE_2D,Te,Ae,fe.width,fe.height)}for(let fe=0,Oe=et.length;fe<Oe;fe++)Me=et[fe],rt?k&&t.texSubImage2D(s.TEXTURE_2D,fe,0,0,Fe,Be,Me):t.texImage2D(s.TEXTURE_2D,fe,Ae,Fe,Be,Me);T.generateMipmaps=!1}else if(rt){if(pt){const fe=zt(pe);t.texStorage2D(s.TEXTURE_2D,Te,Ae,fe.width,fe.height)}k&&t.texSubImage2D(s.TEXTURE_2D,0,0,0,Fe,Be,pe)}else t.texImage2D(s.TEXTURE_2D,0,Ae,Fe,Be,pe);v(T)&&A(he),Pe.__version=ye.version,T.onUpdate&&T.onUpdate(T)}b.__version=T.version}function je(b,T,$){if(T.image.length!==6)return;const he=se(b,T),me=T.source;t.bindTexture(s.TEXTURE_CUBE_MAP,b.__webglTexture,s.TEXTURE0+$);const ye=r.get(me);if(me.version!==ye.__version||he===!0){t.activeTexture(s.TEXTURE0+$);const Pe=xt.getPrimaries(xt.workingColorSpace),ce=T.colorSpace===Pr?null:xt.getPrimaries(T.colorSpace),pe=T.colorSpace===Pr||Pe===ce?s.NONE:s.BROWSER_DEFAULT_WEBGL;t.pixelStorei(s.UNPACK_FLIP_Y_WEBGL,T.flipY),t.pixelStorei(s.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),t.pixelStorei(s.UNPACK_ALIGNMENT,T.unpackAlignment),t.pixelStorei(s.UNPACK_COLORSPACE_CONVERSION_WEBGL,pe);const Fe=T.isCompressedTexture||T.image[0].isCompressedTexture,Be=T.image[0]&&T.image[0].isDataTexture,Ae=[];for(let ge=0;ge<6;ge++)!Fe&&!Be?Ae[ge]=S(T.image[ge],!0,a.maxCubemapSize):Ae[ge]=Be?T.image[ge].image:T.image[ge],Ae[ge]=Ne(T,Ae[ge]);const Me=Ae[0],et=l.convert(T.format,T.colorSpace),rt=l.convert(T.type),pt=L(T.internalFormat,et,rt,T.normalized,T.colorSpace),k=T.isVideoTexture!==!0,Te=ye.__version===void 0||he===!0,fe=me.dataReady;let Oe=D(T,Me);we(s.TEXTURE_CUBE_MAP,T);let Ce;if(Fe){k&&Te&&t.texStorage2D(s.TEXTURE_CUBE_MAP,Oe,pt,Me.width,Me.height);for(let ge=0;ge<6;ge++){Ce=Ae[ge].mipmaps;for(let We=0;We<Ce.length;We++){const st=Ce[We];T.format!==gi?et!==null?k?fe&&t.compressedTexSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,0,0,st.width,st.height,et,st.data):t.compressedTexImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,pt,st.width,st.height,0,st.data):tt("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):k?fe&&t.texSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,0,0,st.width,st.height,et,rt,st.data):t.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,pt,st.width,st.height,0,et,rt,st.data)}}}else{if(Ce=T.mipmaps,k&&Te){Ce.length>0&&Oe++;const ge=zt(Ae[0]);t.texStorage2D(s.TEXTURE_CUBE_MAP,Oe,pt,ge.width,ge.height)}for(let ge=0;ge<6;ge++)if(Be){k?fe&&t.texSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,Ae[ge].width,Ae[ge].height,et,rt,Ae[ge].data):t.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,pt,Ae[ge].width,Ae[ge].height,0,et,rt,Ae[ge].data);for(let We=0;We<Ce.length;We++){const Nt=Ce[We].image[ge].image;k?fe&&t.texSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,0,0,Nt.width,Nt.height,et,rt,Nt.data):t.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,pt,Nt.width,Nt.height,0,et,rt,Nt.data)}}else{k?fe&&t.texSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,et,rt,Ae[ge]):t.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,pt,et,rt,Ae[ge]);for(let We=0;We<Ce.length;We++){const st=Ce[We];k?fe&&t.texSubImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,0,0,et,rt,st.image[ge]):t.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,pt,et,rt,st.image[ge])}}}v(T)&&A(s.TEXTURE_CUBE_MAP),ye.__version=me.version,T.onUpdate&&T.onUpdate(T)}b.__version=T.version}function $e(b,T,$,he,me,ye){const Pe=l.convert($.format,$.colorSpace),ce=l.convert($.type),pe=L($.internalFormat,Pe,ce,$.normalized,$.colorSpace),Fe=r.get(T),Be=r.get($);if(Be.__renderTarget=T,!Fe.__hasExternalTextures){const Ae=Math.max(1,T.width>>ye),Me=Math.max(1,T.height>>ye);me===s.TEXTURE_3D||me===s.TEXTURE_2D_ARRAY?t.texImage3D(me,ye,pe,Ae,Me,T.depth,0,Pe,ce,null):t.texImage2D(me,ye,pe,Ae,Me,0,Pe,ce,null)}t.bindFramebuffer(s.FRAMEBUFFER,b),dt(T)?m.framebufferTexture2DMultisampleEXT(s.FRAMEBUFFER,he,me,Be.__webglTexture,0,Ot(T)):(me===s.TEXTURE_2D||me>=s.TEXTURE_CUBE_MAP_POSITIVE_X&&me<=s.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&s.framebufferTexture2D(s.FRAMEBUFFER,he,me,Be.__webglTexture,ye),t.bindFramebuffer(s.FRAMEBUFFER,null)}function Ut(b,T,$){if(s.bindRenderbuffer(s.RENDERBUFFER,b),T.depthBuffer){const he=T.depthTexture,me=he&&he.isDepthTexture?he.type:null,ye=z(T.stencilBuffer,me),Pe=T.stencilBuffer?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT;dt(T)?m.renderbufferStorageMultisampleEXT(s.RENDERBUFFER,Ot(T),ye,T.width,T.height):$?s.renderbufferStorageMultisample(s.RENDERBUFFER,Ot(T),ye,T.width,T.height):s.renderbufferStorage(s.RENDERBUFFER,ye,T.width,T.height),s.framebufferRenderbuffer(s.FRAMEBUFFER,Pe,s.RENDERBUFFER,b)}else{const he=T.textures;for(let me=0;me<he.length;me++){const ye=he[me],Pe=l.convert(ye.format,ye.colorSpace),ce=l.convert(ye.type),pe=L(ye.internalFormat,Pe,ce,ye.normalized,ye.colorSpace);dt(T)?m.renderbufferStorageMultisampleEXT(s.RENDERBUFFER,Ot(T),pe,T.width,T.height):$?s.renderbufferStorageMultisample(s.RENDERBUFFER,Ot(T),pe,T.width,T.height):s.renderbufferStorage(s.RENDERBUFFER,pe,T.width,T.height)}}s.bindRenderbuffer(s.RENDERBUFFER,null)}function ct(b,T,$){const he=T.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(s.FRAMEBUFFER,b),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const me=r.get(T.depthTexture);if(me.__renderTarget=T,(!me.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),he){if(me.__webglInit===void 0&&(me.__webglInit=!0,T.depthTexture.addEventListener("dispose",F)),me.__webglTexture===void 0){me.__webglTexture=s.createTexture(),t.bindTexture(s.TEXTURE_CUBE_MAP,me.__webglTexture),we(s.TEXTURE_CUBE_MAP,T.depthTexture);const Fe=l.convert(T.depthTexture.format),Be=l.convert(T.depthTexture.type);let Ae;T.depthTexture.format===ir?Ae=s.DEPTH_COMPONENT24:T.depthTexture.format===is&&(Ae=s.DEPTH24_STENCIL8);for(let Me=0;Me<6;Me++)s.texImage2D(s.TEXTURE_CUBE_MAP_POSITIVE_X+Me,0,Ae,T.width,T.height,0,Fe,Be,null)}}else G(T.depthTexture,0);const ye=me.__webglTexture,Pe=Ot(T),ce=he?s.TEXTURE_CUBE_MAP_POSITIVE_X+$:s.TEXTURE_2D,pe=T.depthTexture.format===is?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT;if(T.depthTexture.format===ir)dt(T)?m.framebufferTexture2DMultisampleEXT(s.FRAMEBUFFER,pe,ce,ye,0,Pe):s.framebufferTexture2D(s.FRAMEBUFFER,pe,ce,ye,0);else if(T.depthTexture.format===is)dt(T)?m.framebufferTexture2DMultisampleEXT(s.FRAMEBUFFER,pe,ce,ye,0,Pe):s.framebufferTexture2D(s.FRAMEBUFFER,pe,ce,ye,0);else throw new Error("Unknown depthTexture format")}function Et(b){const T=r.get(b),$=b.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==b.depthTexture){const he=b.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),he){const me=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,he.removeEventListener("dispose",me)};he.addEventListener("dispose",me),T.__depthDisposeCallback=me}T.__boundDepthTexture=he}if(b.depthTexture&&!T.__autoAllocateDepthBuffer)if($)for(let he=0;he<6;he++)ct(T.__webglFramebuffer[he],b,he);else{const he=b.texture.mipmaps;he&&he.length>0?ct(T.__webglFramebuffer[0],b,0):ct(T.__webglFramebuffer,b,0)}else if($){T.__webglDepthbuffer=[];for(let he=0;he<6;he++)if(t.bindFramebuffer(s.FRAMEBUFFER,T.__webglFramebuffer[he]),T.__webglDepthbuffer[he]===void 0)T.__webglDepthbuffer[he]=s.createRenderbuffer(),Ut(T.__webglDepthbuffer[he],b,!1);else{const me=b.stencilBuffer?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT,ye=T.__webglDepthbuffer[he];s.bindRenderbuffer(s.RENDERBUFFER,ye),s.framebufferRenderbuffer(s.FRAMEBUFFER,me,s.RENDERBUFFER,ye)}}else{const he=b.texture.mipmaps;if(he&&he.length>0?t.bindFramebuffer(s.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(s.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=s.createRenderbuffer(),Ut(T.__webglDepthbuffer,b,!1);else{const me=b.stencilBuffer?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT,ye=T.__webglDepthbuffer;s.bindRenderbuffer(s.RENDERBUFFER,ye),s.framebufferRenderbuffer(s.FRAMEBUFFER,me,s.RENDERBUFFER,ye)}}t.bindFramebuffer(s.FRAMEBUFFER,null)}function Dt(b,T,$){const he=r.get(b);T!==void 0&&$e(he.__webglFramebuffer,b,b.texture,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,0),$!==void 0&&Et(b)}function ft(b){const T=b.texture,$=r.get(b),he=r.get(T);b.addEventListener("dispose",R);const me=b.textures,ye=b.isWebGLCubeRenderTarget===!0,Pe=me.length>1;if(Pe||(he.__webglTexture===void 0&&(he.__webglTexture=s.createTexture()),he.__version=T.version,d.memory.textures++),ye){$.__webglFramebuffer=[];for(let ce=0;ce<6;ce++)if(T.mipmaps&&T.mipmaps.length>0){$.__webglFramebuffer[ce]=[];for(let pe=0;pe<T.mipmaps.length;pe++)$.__webglFramebuffer[ce][pe]=s.createFramebuffer()}else $.__webglFramebuffer[ce]=s.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){$.__webglFramebuffer=[];for(let ce=0;ce<T.mipmaps.length;ce++)$.__webglFramebuffer[ce]=s.createFramebuffer()}else $.__webglFramebuffer=s.createFramebuffer();if(Pe)for(let ce=0,pe=me.length;ce<pe;ce++){const Fe=r.get(me[ce]);Fe.__webglTexture===void 0&&(Fe.__webglTexture=s.createTexture(),d.memory.textures++)}if(b.samples>0&&dt(b)===!1){$.__webglMultisampledFramebuffer=s.createFramebuffer(),$.__webglColorRenderbuffer=[],t.bindFramebuffer(s.FRAMEBUFFER,$.__webglMultisampledFramebuffer);for(let ce=0;ce<me.length;ce++){const pe=me[ce];$.__webglColorRenderbuffer[ce]=s.createRenderbuffer(),s.bindRenderbuffer(s.RENDERBUFFER,$.__webglColorRenderbuffer[ce]);const Fe=l.convert(pe.format,pe.colorSpace),Be=l.convert(pe.type),Ae=L(pe.internalFormat,Fe,Be,pe.normalized,pe.colorSpace,b.isXRRenderTarget===!0),Me=Ot(b);s.renderbufferStorageMultisample(s.RENDERBUFFER,Me,Ae,b.width,b.height),s.framebufferRenderbuffer(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0+ce,s.RENDERBUFFER,$.__webglColorRenderbuffer[ce])}s.bindRenderbuffer(s.RENDERBUFFER,null),b.depthBuffer&&($.__webglDepthRenderbuffer=s.createRenderbuffer(),Ut($.__webglDepthRenderbuffer,b,!0)),t.bindFramebuffer(s.FRAMEBUFFER,null)}}if(ye){t.bindTexture(s.TEXTURE_CUBE_MAP,he.__webglTexture),we(s.TEXTURE_CUBE_MAP,T);for(let ce=0;ce<6;ce++)if(T.mipmaps&&T.mipmaps.length>0)for(let pe=0;pe<T.mipmaps.length;pe++)$e($.__webglFramebuffer[ce][pe],b,T,s.COLOR_ATTACHMENT0,s.TEXTURE_CUBE_MAP_POSITIVE_X+ce,pe);else $e($.__webglFramebuffer[ce],b,T,s.COLOR_ATTACHMENT0,s.TEXTURE_CUBE_MAP_POSITIVE_X+ce,0);v(T)&&A(s.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Pe){for(let ce=0,pe=me.length;ce<pe;ce++){const Fe=me[ce],Be=r.get(Fe);let Ae=s.TEXTURE_2D;(b.isWebGL3DRenderTarget||b.isWebGLArrayRenderTarget)&&(Ae=b.isWebGL3DRenderTarget?s.TEXTURE_3D:s.TEXTURE_2D_ARRAY),t.bindTexture(Ae,Be.__webglTexture),we(Ae,Fe),$e($.__webglFramebuffer,b,Fe,s.COLOR_ATTACHMENT0+ce,Ae,0),v(Fe)&&A(Ae)}t.unbindTexture()}else{let ce=s.TEXTURE_2D;if((b.isWebGL3DRenderTarget||b.isWebGLArrayRenderTarget)&&(ce=b.isWebGL3DRenderTarget?s.TEXTURE_3D:s.TEXTURE_2D_ARRAY),t.bindTexture(ce,he.__webglTexture),we(ce,T),T.mipmaps&&T.mipmaps.length>0)for(let pe=0;pe<T.mipmaps.length;pe++)$e($.__webglFramebuffer[pe],b,T,s.COLOR_ATTACHMENT0,ce,pe);else $e($.__webglFramebuffer,b,T,s.COLOR_ATTACHMENT0,ce,0);v(T)&&A(ce),t.unbindTexture()}b.depthBuffer&&Et(b)}function Yt(b){const T=b.textures;for(let $=0,he=T.length;$<he;$++){const me=T[$];if(v(me)){const ye=P(b),Pe=r.get(me).__webglTexture;t.bindTexture(ye,Pe),A(ye),t.unbindTexture()}}}const Ft=[],hn=[];function H(b){if(b.samples>0){if(dt(b)===!1){const T=b.textures,$=b.width,he=b.height;let me=s.COLOR_BUFFER_BIT;const ye=b.stencilBuffer?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT,Pe=r.get(b),ce=T.length>1;if(ce)for(let Fe=0;Fe<T.length;Fe++)t.bindFramebuffer(s.FRAMEBUFFER,Pe.__webglMultisampledFramebuffer),s.framebufferRenderbuffer(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0+Fe,s.RENDERBUFFER,null),t.bindFramebuffer(s.FRAMEBUFFER,Pe.__webglFramebuffer),s.framebufferTexture2D(s.DRAW_FRAMEBUFFER,s.COLOR_ATTACHMENT0+Fe,s.TEXTURE_2D,null,0);t.bindFramebuffer(s.READ_FRAMEBUFFER,Pe.__webglMultisampledFramebuffer);const pe=b.texture.mipmaps;pe&&pe.length>0?t.bindFramebuffer(s.DRAW_FRAMEBUFFER,Pe.__webglFramebuffer[0]):t.bindFramebuffer(s.DRAW_FRAMEBUFFER,Pe.__webglFramebuffer);for(let Fe=0;Fe<T.length;Fe++){if(b.resolveDepthBuffer&&(b.depthBuffer&&(me|=s.DEPTH_BUFFER_BIT),b.stencilBuffer&&b.resolveStencilBuffer&&(me|=s.STENCIL_BUFFER_BIT)),ce){s.framebufferRenderbuffer(s.READ_FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.RENDERBUFFER,Pe.__webglColorRenderbuffer[Fe]);const Be=r.get(T[Fe]).__webglTexture;s.framebufferTexture2D(s.DRAW_FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,Be,0)}s.blitFramebuffer(0,0,$,he,0,0,$,he,me,s.NEAREST),g===!0&&(Ft.length=0,hn.length=0,Ft.push(s.COLOR_ATTACHMENT0+Fe),b.depthBuffer&&b.resolveDepthBuffer===!1&&(Ft.push(ye),hn.push(ye),s.invalidateFramebuffer(s.DRAW_FRAMEBUFFER,hn)),s.invalidateFramebuffer(s.READ_FRAMEBUFFER,Ft))}if(t.bindFramebuffer(s.READ_FRAMEBUFFER,null),t.bindFramebuffer(s.DRAW_FRAMEBUFFER,null),ce)for(let Fe=0;Fe<T.length;Fe++){t.bindFramebuffer(s.FRAMEBUFFER,Pe.__webglMultisampledFramebuffer),s.framebufferRenderbuffer(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0+Fe,s.RENDERBUFFER,Pe.__webglColorRenderbuffer[Fe]);const Be=r.get(T[Fe]).__webglTexture;t.bindFramebuffer(s.FRAMEBUFFER,Pe.__webglFramebuffer),s.framebufferTexture2D(s.DRAW_FRAMEBUFFER,s.COLOR_ATTACHMENT0+Fe,s.TEXTURE_2D,Be,0)}t.bindFramebuffer(s.DRAW_FRAMEBUFFER,Pe.__webglMultisampledFramebuffer)}else if(b.depthBuffer&&b.resolveDepthBuffer===!1&&g){const T=b.stencilBuffer?s.DEPTH_STENCIL_ATTACHMENT:s.DEPTH_ATTACHMENT;s.invalidateFramebuffer(s.DRAW_FRAMEBUFFER,[T])}}}function Ot(b){return Math.min(a.maxSamples,b.samples)}function dt(b){const T=r.get(b);return b.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function Ct(b){const T=d.render.frame;M.get(b)!==T&&(M.set(b,T),b.update())}function Ne(b,T){const $=b.colorSpace,he=b.format,me=b.type;return b.isCompressedTexture===!0||b.isVideoTexture===!0||$!==jl&&$!==Pr&&(xt.getTransfer($)===Lt?(he!==gi||me!==ii)&&tt("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Mt("WebGLTextures: Unsupported texture color space:",$)),T}function zt(b){return typeof HTMLImageElement<"u"&&b instanceof HTMLImageElement?(_.width=b.naturalWidth||b.width,_.height=b.naturalHeight||b.height):typeof VideoFrame<"u"&&b instanceof VideoFrame?(_.width=b.displayWidth,_.height=b.displayHeight):(_.width=b.width,_.height=b.height),_}this.allocateTextureUnit=Z,this.resetTextureUnits=re,this.getTextureUnits=ae,this.setTextureUnits=X,this.setTexture2D=G,this.setTexture2DArray=J,this.setTexture3D=ie,this.setTextureCube=U,this.rebindTextures=Dt,this.setupRenderTarget=ft,this.updateRenderTargetMipmap=Yt,this.updateMultisampleRenderTarget=H,this.setupDepthRenderbuffer=Et,this.setupFrameBufferTexture=$e,this.useMultisampledRTT=dt,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function $E(s,e){function t(r,a=Pr){let l;const d=xt.getTransfer(a);if(r===ii)return s.UNSIGNED_BYTE;if(r===_d)return s.UNSIGNED_SHORT_4_4_4_4;if(r===gd)return s.UNSIGNED_SHORT_5_5_5_1;if(r===v_)return s.UNSIGNED_INT_5_9_9_9_REV;if(r===x_)return s.UNSIGNED_INT_10F_11F_11F_REV;if(r===__)return s.BYTE;if(r===g_)return s.SHORT;if(r===na)return s.UNSIGNED_SHORT;if(r===md)return s.INT;if(r===Ni)return s.UNSIGNED_INT;if(r===Pi)return s.FLOAT;if(r===nr)return s.HALF_FLOAT;if(r===S_)return s.ALPHA;if(r===y_)return s.RGB;if(r===gi)return s.RGBA;if(r===ir)return s.DEPTH_COMPONENT;if(r===is)return s.DEPTH_STENCIL;if(r===M_)return s.RED;if(r===vd)return s.RED_INTEGER;if(r===os)return s.RG;if(r===xd)return s.RG_INTEGER;if(r===Sd)return s.RGBA_INTEGER;if(r===zl||r===Hl||r===Vl||r===Gl)if(d===Lt)if(l=e.get("WEBGL_compressed_texture_s3tc_srgb"),l!==null){if(r===zl)return l.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(r===Hl)return l.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(r===Vl)return l.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(r===Gl)return l.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(l=e.get("WEBGL_compressed_texture_s3tc"),l!==null){if(r===zl)return l.COMPRESSED_RGB_S3TC_DXT1_EXT;if(r===Hl)return l.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(r===Vl)return l.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(r===Gl)return l.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(r===If||r===Nf||r===Uf||r===Ff)if(l=e.get("WEBGL_compressed_texture_pvrtc"),l!==null){if(r===If)return l.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(r===Nf)return l.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(r===Uf)return l.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(r===Ff)return l.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(r===Of||r===Bf||r===kf||r===zf||r===Hf||r===Yl||r===Vf)if(l=e.get("WEBGL_compressed_texture_etc"),l!==null){if(r===Of||r===Bf)return d===Lt?l.COMPRESSED_SRGB8_ETC2:l.COMPRESSED_RGB8_ETC2;if(r===kf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:l.COMPRESSED_RGBA8_ETC2_EAC;if(r===zf)return l.COMPRESSED_R11_EAC;if(r===Hf)return l.COMPRESSED_SIGNED_R11_EAC;if(r===Yl)return l.COMPRESSED_RG11_EAC;if(r===Vf)return l.COMPRESSED_SIGNED_RG11_EAC}else return null;if(r===Gf||r===Wf||r===Xf||r===Yf||r===qf||r===jf||r===Kf||r===$f||r===Zf||r===Qf||r===Jf||r===ed||r===td||r===nd)if(l=e.get("WEBGL_compressed_texture_astc"),l!==null){if(r===Gf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:l.COMPRESSED_RGBA_ASTC_4x4_KHR;if(r===Wf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:l.COMPRESSED_RGBA_ASTC_5x4_KHR;if(r===Xf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:l.COMPRESSED_RGBA_ASTC_5x5_KHR;if(r===Yf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:l.COMPRESSED_RGBA_ASTC_6x5_KHR;if(r===qf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:l.COMPRESSED_RGBA_ASTC_6x6_KHR;if(r===jf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:l.COMPRESSED_RGBA_ASTC_8x5_KHR;if(r===Kf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:l.COMPRESSED_RGBA_ASTC_8x6_KHR;if(r===$f)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:l.COMPRESSED_RGBA_ASTC_8x8_KHR;if(r===Zf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:l.COMPRESSED_RGBA_ASTC_10x5_KHR;if(r===Qf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:l.COMPRESSED_RGBA_ASTC_10x6_KHR;if(r===Jf)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:l.COMPRESSED_RGBA_ASTC_10x8_KHR;if(r===ed)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:l.COMPRESSED_RGBA_ASTC_10x10_KHR;if(r===td)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:l.COMPRESSED_RGBA_ASTC_12x10_KHR;if(r===nd)return d===Lt?l.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:l.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(r===id||r===rd||r===sd)if(l=e.get("EXT_texture_compression_bptc"),l!==null){if(r===id)return d===Lt?l.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:l.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(r===rd)return l.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(r===sd)return l.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(r===od||r===ad||r===ql||r===ld)if(l=e.get("EXT_texture_compression_rgtc"),l!==null){if(r===od)return l.COMPRESSED_RED_RGTC1_EXT;if(r===ad)return l.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(r===ql)return l.COMPRESSED_RED_GREEN_RGTC2_EXT;if(r===ld)return l.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return r===ia?s.UNSIGNED_INT_24_8:s[r]!==void 0?s[r]:null}return{convert:t}}const ZE=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,QE=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class JE{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const r=new N_(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=r}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,r=new xi({vertexShader:ZE,fragmentShader:QE,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new rr(new tu(20,20),r)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class eT extends ls{constructor(e,t){super();const r=this;let a=null,l=1,d=null,m="local-floor",g=1,_=null,M=null,u=null,f=null,p=null,y=null;const E=typeof XRWebGLBinding<"u",S=new JE,v={},A=t.getContextAttributes();let P=null,L=null;const z=[],D=[],F=new It;let R=null;const I=new ni;I.viewport=new Jt;const W=new ni;W.viewport=new Jt;const O=[I,W],j=new cx;let re=null,ae=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(se){let _e=z[se];return _e===void 0&&(_e=new Kc,z[se]=_e),_e.getTargetRaySpace()},this.getControllerGrip=function(se){let _e=z[se];return _e===void 0&&(_e=new Kc,z[se]=_e),_e.getGripSpace()},this.getHand=function(se){let _e=z[se];return _e===void 0&&(_e=new Kc,z[se]=_e),_e.getHandSpace()};function X(se){const _e=D.indexOf(se.inputSource);if(_e===-1)return;const de=z[_e];de!==void 0&&(de.update(se.inputSource,se.frame,_||d),de.dispatchEvent({type:se.type,data:se.inputSource}))}function Z(){a.removeEventListener("select",X),a.removeEventListener("selectstart",X),a.removeEventListener("selectend",X),a.removeEventListener("squeeze",X),a.removeEventListener("squeezestart",X),a.removeEventListener("squeezeend",X),a.removeEventListener("end",Z),a.removeEventListener("inputsourceschange",q);for(let se=0;se<z.length;se++){const _e=D[se];_e!==null&&(D[se]=null,z[se].disconnect(_e))}re=null,ae=null,S.reset();for(const se in v)delete v[se];e.setRenderTarget(P),p=null,f=null,u=null,a=null,L=null,we.stop(),r.isPresenting=!1,e.setPixelRatio(R),e.setSize(F.width,F.height,!1),r.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(se){l=se,r.isPresenting===!0&&tt("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(se){m=se,r.isPresenting===!0&&tt("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return _||d},this.setReferenceSpace=function(se){_=se},this.getBaseLayer=function(){return f!==null?f:p},this.getBinding=function(){return u===null&&E&&(u=new XRWebGLBinding(a,t)),u},this.getFrame=function(){return y},this.getSession=function(){return a},this.setSession=async function(se){if(a=se,a!==null){if(P=e.getRenderTarget(),a.addEventListener("select",X),a.addEventListener("selectstart",X),a.addEventListener("selectend",X),a.addEventListener("squeeze",X),a.addEventListener("squeezestart",X),a.addEventListener("squeezeend",X),a.addEventListener("end",Z),a.addEventListener("inputsourceschange",q),A.xrCompatible!==!0&&await t.makeXRCompatible(),R=e.getPixelRatio(),e.getSize(F),E&&"createProjectionLayer"in XRWebGLBinding.prototype){let de=null,Ie=null,je=null;A.depth&&(je=A.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,de=A.stencil?is:ir,Ie=A.stencil?ia:Ni);const $e={colorFormat:t.RGBA8,depthFormat:je,scaleFactor:l};u=this.getBinding(),f=u.createProjectionLayer($e),a.updateRenderState({layers:[f]}),e.setPixelRatio(1),e.setSize(f.textureWidth,f.textureHeight,!1),L=new Ii(f.textureWidth,f.textureHeight,{format:gi,type:ii,depthTexture:new Js(f.textureWidth,f.textureHeight,Ie,void 0,void 0,void 0,void 0,void 0,void 0,de),stencilBuffer:A.stencil,colorSpace:e.outputColorSpace,samples:A.antialias?4:0,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}else{const de={antialias:A.antialias,alpha:!0,depth:A.depth,stencil:A.stencil,framebufferScaleFactor:l};p=new XRWebGLLayer(a,t,de),a.updateRenderState({baseLayer:p}),e.setPixelRatio(1),e.setSize(p.framebufferWidth,p.framebufferHeight,!1),L=new Ii(p.framebufferWidth,p.framebufferHeight,{format:gi,type:ii,colorSpace:e.outputColorSpace,stencilBuffer:A.stencil,resolveDepthBuffer:p.ignoreDepthValues===!1,resolveStencilBuffer:p.ignoreDepthValues===!1})}L.isXRRenderTarget=!0,this.setFoveation(g),_=null,d=await a.requestReferenceSpace(m),we.setContext(a),we.start(),r.isPresenting=!0,r.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(a!==null)return a.environmentBlendMode},this.getDepthTexture=function(){return S.getDepthTexture()};function q(se){for(let _e=0;_e<se.removed.length;_e++){const de=se.removed[_e],Ie=D.indexOf(de);Ie>=0&&(D[Ie]=null,z[Ie].disconnect(de))}for(let _e=0;_e<se.added.length;_e++){const de=se.added[_e];let Ie=D.indexOf(de);if(Ie===-1){for(let $e=0;$e<z.length;$e++)if($e>=D.length){D.push(de),Ie=$e;break}else if(D[$e]===null){D[$e]=de,Ie=$e;break}if(Ie===-1)break}const je=z[Ie];je&&je.connect(de)}}const G=new oe,J=new oe;function ie(se,_e,de){G.setFromMatrixPosition(_e.matrixWorld),J.setFromMatrixPosition(de.matrixWorld);const Ie=G.distanceTo(J),je=_e.projectionMatrix.elements,$e=de.projectionMatrix.elements,Ut=je[14]/(je[10]-1),ct=je[14]/(je[10]+1),Et=(je[9]+1)/je[5],Dt=(je[9]-1)/je[5],ft=(je[8]-1)/je[0],Yt=($e[8]+1)/$e[0],Ft=Ut*ft,hn=Ut*Yt,H=Ie/(-ft+Yt),Ot=H*-ft;if(_e.matrixWorld.decompose(se.position,se.quaternion,se.scale),se.translateX(Ot),se.translateZ(H),se.matrixWorld.compose(se.position,se.quaternion,se.scale),se.matrixWorldInverse.copy(se.matrixWorld).invert(),je[10]===-1)se.projectionMatrix.copy(_e.projectionMatrix),se.projectionMatrixInverse.copy(_e.projectionMatrixInverse);else{const dt=Ut+H,Ct=ct+H,Ne=Ft-Ot,zt=hn+(Ie-Ot),b=Et*ct/Ct*dt,T=Dt*ct/Ct*dt;se.projectionMatrix.makePerspective(Ne,zt,b,T,dt,Ct),se.projectionMatrixInverse.copy(se.projectionMatrix).invert()}}function U(se,_e){_e===null?se.matrixWorld.copy(se.matrix):se.matrixWorld.multiplyMatrices(_e.matrixWorld,se.matrix),se.matrixWorldInverse.copy(se.matrixWorld).invert()}this.updateCamera=function(se){if(a===null)return;let _e=se.near,de=se.far;S.texture!==null&&(S.depthNear>0&&(_e=S.depthNear),S.depthFar>0&&(de=S.depthFar)),j.near=W.near=I.near=_e,j.far=W.far=I.far=de,(re!==j.near||ae!==j.far)&&(a.updateRenderState({depthNear:j.near,depthFar:j.far}),re=j.near,ae=j.far),j.layers.mask=se.layers.mask|6,I.layers.mask=j.layers.mask&-5,W.layers.mask=j.layers.mask&-3;const Ie=se.parent,je=j.cameras;U(j,Ie);for(let $e=0;$e<je.length;$e++)U(je[$e],Ie);je.length===2?ie(j,I,W):j.projectionMatrix.copy(I.projectionMatrix),K(se,j,Ie)};function K(se,_e,de){de===null?se.matrix.copy(_e.matrixWorld):(se.matrix.copy(de.matrixWorld),se.matrix.invert(),se.matrix.multiply(_e.matrixWorld)),se.matrix.decompose(se.position,se.quaternion,se.scale),se.updateMatrixWorld(!0),se.projectionMatrix.copy(_e.projectionMatrix),se.projectionMatrixInverse.copy(_e.projectionMatrixInverse),se.isPerspectiveCamera&&(se.fov=ra*2*Math.atan(1/se.projectionMatrix.elements[5]),se.zoom=1)}this.getCamera=function(){return j},this.getFoveation=function(){if(!(f===null&&p===null))return g},this.setFoveation=function(se){g=se,f!==null&&(f.fixedFoveation=se),p!==null&&p.fixedFoveation!==void 0&&(p.fixedFoveation=se)},this.hasDepthSensing=function(){return S.texture!==null},this.getDepthSensingMesh=function(){return S.getMesh(j)},this.getCameraTexture=function(se){return v[se]};let Le=null;function De(se,_e){if(M=_e.getViewerPose(_||d),y=_e,M!==null){const de=M.views;p!==null&&(e.setRenderTargetFramebuffer(L,p.framebuffer),e.setRenderTarget(L));let Ie=!1;de.length!==j.cameras.length&&(j.cameras.length=0,Ie=!0);for(let ct=0;ct<de.length;ct++){const Et=de[ct];let Dt=null;if(p!==null)Dt=p.getViewport(Et);else{const Yt=u.getViewSubImage(f,Et);Dt=Yt.viewport,ct===0&&(e.setRenderTargetTextures(L,Yt.colorTexture,Yt.depthStencilTexture),e.setRenderTarget(L))}let ft=O[ct];ft===void 0&&(ft=new ni,ft.layers.enable(ct),ft.viewport=new Jt,O[ct]=ft),ft.matrix.fromArray(Et.transform.matrix),ft.matrix.decompose(ft.position,ft.quaternion,ft.scale),ft.projectionMatrix.fromArray(Et.projectionMatrix),ft.projectionMatrixInverse.copy(ft.projectionMatrix).invert(),ft.viewport.set(Dt.x,Dt.y,Dt.width,Dt.height),ct===0&&(j.matrix.copy(ft.matrix),j.matrix.decompose(j.position,j.quaternion,j.scale)),Ie===!0&&j.cameras.push(ft)}const je=a.enabledFeatures;if(je&&je.includes("depth-sensing")&&a.depthUsage=="gpu-optimized"&&E){u=r.getBinding();const ct=u.getDepthInformation(de[0]);ct&&ct.isValid&&ct.texture&&S.init(ct,a.renderState)}if(je&&je.includes("camera-access")&&E){e.state.unbindTexture(),u=r.getBinding();for(let ct=0;ct<de.length;ct++){const Et=de[ct].camera;if(Et){let Dt=v[Et];Dt||(Dt=new N_,v[Et]=Dt);const ft=u.getCameraImage(Et);Dt.sourceTexture=ft}}}}for(let de=0;de<z.length;de++){const Ie=D[de],je=z[de];Ie!==null&&je!==void 0&&je.update(Ie,_e,_||d)}Le&&Le(se,_e),_e.detectedPlanes&&r.dispatchEvent({type:"planesdetected",data:_e}),y=null}const we=new B_;we.setAnimationLoop(De),this.setAnimationLoop=function(se){Le=se},this.dispose=function(){}}}const tT=new rn,X_=new lt;X_.set(-1,0,0,0,1,0,0,0,1);function nT(s,e){function t(S,v){S.matrixAutoUpdate===!0&&S.updateMatrix(),v.value.copy(S.matrix)}function r(S,v){v.color.getRGB(S.fogColor.value,U_(s)),v.isFog?(S.fogNear.value=v.near,S.fogFar.value=v.far):v.isFogExp2&&(S.fogDensity.value=v.density)}function a(S,v,A,P,L){v.isNodeMaterial?v.uniformsNeedUpdate=!1:v.isMeshBasicMaterial?l(S,v):v.isMeshLambertMaterial?(l(S,v),v.envMap&&(S.envMapIntensity.value=v.envMapIntensity)):v.isMeshToonMaterial?(l(S,v),u(S,v)):v.isMeshPhongMaterial?(l(S,v),M(S,v),v.envMap&&(S.envMapIntensity.value=v.envMapIntensity)):v.isMeshStandardMaterial?(l(S,v),f(S,v),v.isMeshPhysicalMaterial&&p(S,v,L)):v.isMeshMatcapMaterial?(l(S,v),y(S,v)):v.isMeshDepthMaterial?l(S,v):v.isMeshDistanceMaterial?(l(S,v),E(S,v)):v.isMeshNormalMaterial?l(S,v):v.isLineBasicMaterial?(d(S,v),v.isLineDashedMaterial&&m(S,v)):v.isPointsMaterial?g(S,v,A,P):v.isSpriteMaterial?_(S,v):v.isShadowMaterial?(S.color.value.copy(v.color),S.opacity.value=v.opacity):v.isShaderMaterial&&(v.uniformsNeedUpdate=!1)}function l(S,v){S.opacity.value=v.opacity,v.color&&S.diffuse.value.copy(v.color),v.emissive&&S.emissive.value.copy(v.emissive).multiplyScalar(v.emissiveIntensity),v.map&&(S.map.value=v.map,t(v.map,S.mapTransform)),v.alphaMap&&(S.alphaMap.value=v.alphaMap,t(v.alphaMap,S.alphaMapTransform)),v.bumpMap&&(S.bumpMap.value=v.bumpMap,t(v.bumpMap,S.bumpMapTransform),S.bumpScale.value=v.bumpScale,v.side===kn&&(S.bumpScale.value*=-1)),v.normalMap&&(S.normalMap.value=v.normalMap,t(v.normalMap,S.normalMapTransform),S.normalScale.value.copy(v.normalScale),v.side===kn&&S.normalScale.value.negate()),v.displacementMap&&(S.displacementMap.value=v.displacementMap,t(v.displacementMap,S.displacementMapTransform),S.displacementScale.value=v.displacementScale,S.displacementBias.value=v.displacementBias),v.emissiveMap&&(S.emissiveMap.value=v.emissiveMap,t(v.emissiveMap,S.emissiveMapTransform)),v.specularMap&&(S.specularMap.value=v.specularMap,t(v.specularMap,S.specularMapTransform)),v.alphaTest>0&&(S.alphaTest.value=v.alphaTest);const A=e.get(v),P=A.envMap,L=A.envMapRotation;P&&(S.envMap.value=P,S.envMapRotation.value.setFromMatrix4(tT.makeRotationFromEuler(L)).transpose(),P.isCubeTexture&&P.isRenderTargetTexture===!1&&S.envMapRotation.value.premultiply(X_),S.reflectivity.value=v.reflectivity,S.ior.value=v.ior,S.refractionRatio.value=v.refractionRatio),v.lightMap&&(S.lightMap.value=v.lightMap,S.lightMapIntensity.value=v.lightMapIntensity,t(v.lightMap,S.lightMapTransform)),v.aoMap&&(S.aoMap.value=v.aoMap,S.aoMapIntensity.value=v.aoMapIntensity,t(v.aoMap,S.aoMapTransform))}function d(S,v){S.diffuse.value.copy(v.color),S.opacity.value=v.opacity,v.map&&(S.map.value=v.map,t(v.map,S.mapTransform))}function m(S,v){S.dashSize.value=v.dashSize,S.totalSize.value=v.dashSize+v.gapSize,S.scale.value=v.scale}function g(S,v,A,P){S.diffuse.value.copy(v.color),S.opacity.value=v.opacity,S.size.value=v.size*A,S.scale.value=P*.5,v.map&&(S.map.value=v.map,t(v.map,S.uvTransform)),v.alphaMap&&(S.alphaMap.value=v.alphaMap,t(v.alphaMap,S.alphaMapTransform)),v.alphaTest>0&&(S.alphaTest.value=v.alphaTest)}function _(S,v){S.diffuse.value.copy(v.color),S.opacity.value=v.opacity,S.rotation.value=v.rotation,v.map&&(S.map.value=v.map,t(v.map,S.mapTransform)),v.alphaMap&&(S.alphaMap.value=v.alphaMap,t(v.alphaMap,S.alphaMapTransform)),v.alphaTest>0&&(S.alphaTest.value=v.alphaTest)}function M(S,v){S.specular.value.copy(v.specular),S.shininess.value=Math.max(v.shininess,1e-4)}function u(S,v){v.gradientMap&&(S.gradientMap.value=v.gradientMap)}function f(S,v){S.metalness.value=v.metalness,v.metalnessMap&&(S.metalnessMap.value=v.metalnessMap,t(v.metalnessMap,S.metalnessMapTransform)),S.roughness.value=v.roughness,v.roughnessMap&&(S.roughnessMap.value=v.roughnessMap,t(v.roughnessMap,S.roughnessMapTransform)),v.envMap&&(S.envMapIntensity.value=v.envMapIntensity)}function p(S,v,A){S.ior.value=v.ior,v.sheen>0&&(S.sheenColor.value.copy(v.sheenColor).multiplyScalar(v.sheen),S.sheenRoughness.value=v.sheenRoughness,v.sheenColorMap&&(S.sheenColorMap.value=v.sheenColorMap,t(v.sheenColorMap,S.sheenColorMapTransform)),v.sheenRoughnessMap&&(S.sheenRoughnessMap.value=v.sheenRoughnessMap,t(v.sheenRoughnessMap,S.sheenRoughnessMapTransform))),v.clearcoat>0&&(S.clearcoat.value=v.clearcoat,S.clearcoatRoughness.value=v.clearcoatRoughness,v.clearcoatMap&&(S.clearcoatMap.value=v.clearcoatMap,t(v.clearcoatMap,S.clearcoatMapTransform)),v.clearcoatRoughnessMap&&(S.clearcoatRoughnessMap.value=v.clearcoatRoughnessMap,t(v.clearcoatRoughnessMap,S.clearcoatRoughnessMapTransform)),v.clearcoatNormalMap&&(S.clearcoatNormalMap.value=v.clearcoatNormalMap,t(v.clearcoatNormalMap,S.clearcoatNormalMapTransform),S.clearcoatNormalScale.value.copy(v.clearcoatNormalScale),v.side===kn&&S.clearcoatNormalScale.value.negate())),v.dispersion>0&&(S.dispersion.value=v.dispersion),v.iridescence>0&&(S.iridescence.value=v.iridescence,S.iridescenceIOR.value=v.iridescenceIOR,S.iridescenceThicknessMinimum.value=v.iridescenceThicknessRange[0],S.iridescenceThicknessMaximum.value=v.iridescenceThicknessRange[1],v.iridescenceMap&&(S.iridescenceMap.value=v.iridescenceMap,t(v.iridescenceMap,S.iridescenceMapTransform)),v.iridescenceThicknessMap&&(S.iridescenceThicknessMap.value=v.iridescenceThicknessMap,t(v.iridescenceThicknessMap,S.iridescenceThicknessMapTransform))),v.transmission>0&&(S.transmission.value=v.transmission,S.transmissionSamplerMap.value=A.texture,S.transmissionSamplerSize.value.set(A.width,A.height),v.transmissionMap&&(S.transmissionMap.value=v.transmissionMap,t(v.transmissionMap,S.transmissionMapTransform)),S.thickness.value=v.thickness,v.thicknessMap&&(S.thicknessMap.value=v.thicknessMap,t(v.thicknessMap,S.thicknessMapTransform)),S.attenuationDistance.value=v.attenuationDistance,S.attenuationColor.value.copy(v.attenuationColor)),v.anisotropy>0&&(S.anisotropyVector.value.set(v.anisotropy*Math.cos(v.anisotropyRotation),v.anisotropy*Math.sin(v.anisotropyRotation)),v.anisotropyMap&&(S.anisotropyMap.value=v.anisotropyMap,t(v.anisotropyMap,S.anisotropyMapTransform))),S.specularIntensity.value=v.specularIntensity,S.specularColor.value.copy(v.specularColor),v.specularColorMap&&(S.specularColorMap.value=v.specularColorMap,t(v.specularColorMap,S.specularColorMapTransform)),v.specularIntensityMap&&(S.specularIntensityMap.value=v.specularIntensityMap,t(v.specularIntensityMap,S.specularIntensityMapTransform))}function y(S,v){v.matcap&&(S.matcap.value=v.matcap)}function E(S,v){const A=e.get(v).light;S.referencePosition.value.setFromMatrixPosition(A.matrixWorld),S.nearDistance.value=A.shadow.camera.near,S.farDistance.value=A.shadow.camera.far}return{refreshFogUniforms:r,refreshMaterialUniforms:a}}function iT(s,e,t,r){let a={},l={},d=[];const m=s.getParameter(s.MAX_UNIFORM_BUFFER_BINDINGS);function g(A,P){const L=P.program;r.uniformBlockBinding(A,L)}function _(A,P){let L=a[A.id];L===void 0&&(y(A),L=M(A),a[A.id]=L,A.addEventListener("dispose",S));const z=P.program;r.updateUBOMapping(A,z);const D=e.render.frame;l[A.id]!==D&&(f(A),l[A.id]=D)}function M(A){const P=u();A.__bindingPointIndex=P;const L=s.createBuffer(),z=A.__size,D=A.usage;return s.bindBuffer(s.UNIFORM_BUFFER,L),s.bufferData(s.UNIFORM_BUFFER,z,D),s.bindBuffer(s.UNIFORM_BUFFER,null),s.bindBufferBase(s.UNIFORM_BUFFER,P,L),L}function u(){for(let A=0;A<m;A++)if(d.indexOf(A)===-1)return d.push(A),A;return Mt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function f(A){const P=a[A.id],L=A.uniforms,z=A.__cache;s.bindBuffer(s.UNIFORM_BUFFER,P);for(let D=0,F=L.length;D<F;D++){const R=Array.isArray(L[D])?L[D]:[L[D]];for(let I=0,W=R.length;I<W;I++){const O=R[I];if(p(O,D,I,z)===!0){const j=O.__offset,re=Array.isArray(O.value)?O.value:[O.value];let ae=0;for(let X=0;X<re.length;X++){const Z=re[X],q=E(Z);typeof Z=="number"||typeof Z=="boolean"?(O.__data[0]=Z,s.bufferSubData(s.UNIFORM_BUFFER,j+ae,O.__data)):Z.isMatrix3?(O.__data[0]=Z.elements[0],O.__data[1]=Z.elements[1],O.__data[2]=Z.elements[2],O.__data[3]=0,O.__data[4]=Z.elements[3],O.__data[5]=Z.elements[4],O.__data[6]=Z.elements[5],O.__data[7]=0,O.__data[8]=Z.elements[6],O.__data[9]=Z.elements[7],O.__data[10]=Z.elements[8],O.__data[11]=0):ArrayBuffer.isView(Z)?O.__data.set(new Z.constructor(Z.buffer,Z.byteOffset,O.__data.length)):(Z.toArray(O.__data,ae),ae+=q.storage/Float32Array.BYTES_PER_ELEMENT)}s.bufferSubData(s.UNIFORM_BUFFER,j,O.__data)}}}s.bindBuffer(s.UNIFORM_BUFFER,null)}function p(A,P,L,z){const D=A.value,F=P+"_"+L;if(z[F]===void 0)return typeof D=="number"||typeof D=="boolean"?z[F]=D:ArrayBuffer.isView(D)?z[F]=D.slice():z[F]=D.clone(),!0;{const R=z[F];if(typeof D=="number"||typeof D=="boolean"){if(R!==D)return z[F]=D,!0}else{if(ArrayBuffer.isView(D))return!0;if(R.equals(D)===!1)return R.copy(D),!0}}return!1}function y(A){const P=A.uniforms;let L=0;const z=16;for(let F=0,R=P.length;F<R;F++){const I=Array.isArray(P[F])?P[F]:[P[F]];for(let W=0,O=I.length;W<O;W++){const j=I[W],re=Array.isArray(j.value)?j.value:[j.value];for(let ae=0,X=re.length;ae<X;ae++){const Z=re[ae],q=E(Z),G=L%z,J=G%q.boundary,ie=G+J;L+=J,ie!==0&&z-ie<q.storage&&(L+=z-ie),j.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),j.__offset=L,L+=q.storage}}}const D=L%z;return D>0&&(L+=z-D),A.__size=L,A.__cache={},this}function E(A){const P={boundary:0,storage:0};return typeof A=="number"||typeof A=="boolean"?(P.boundary=4,P.storage=4):A.isVector2?(P.boundary=8,P.storage=8):A.isVector3||A.isColor?(P.boundary=16,P.storage=12):A.isVector4?(P.boundary=16,P.storage=16):A.isMatrix3?(P.boundary=48,P.storage=48):A.isMatrix4?(P.boundary=64,P.storage=64):A.isTexture?tt("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(A)?(P.boundary=16,P.storage=A.byteLength):tt("WebGLRenderer: Unsupported uniform value type.",A),P}function S(A){const P=A.target;P.removeEventListener("dispose",S);const L=d.indexOf(P.__bindingPointIndex);d.splice(L,1),s.deleteBuffer(a[P.id]),delete a[P.id],delete l[P.id]}function v(){for(const A in a)s.deleteBuffer(a[A]);d=[],a={},l={}}return{bind:g,update:_,dispose:v}}const rT=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]);let Ri=null;function sT(){return Ri===null&&(Ri=new Zv(rT,16,16,os,nr),Ri.name="DFG_LUT",Ri.minFilter=Tn,Ri.magFilter=Tn,Ri.wrapS=Qi,Ri.wrapT=Qi,Ri.generateMipmaps=!1,Ri.needsUpdate=!0),Ri}class oT{constructor(e={}){const{canvas:t=dv(),context:r=null,depth:a=!0,stencil:l=!1,alpha:d=!1,antialias:m=!1,premultipliedAlpha:g=!0,preserveDrawingBuffer:_=!1,powerPreference:M="default",failIfMajorPerformanceCaveat:u=!1,reversedDepthBuffer:f=!1,outputBufferType:p=ii}=e;this.isWebGLRenderer=!0;let y;if(r!==null){if(typeof WebGLRenderingContext<"u"&&r instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");y=r.getContextAttributes().alpha}else y=d;const E=p,S=new Set([Sd,xd,vd]),v=new Set([ii,Ni,na,ia,_d,gd]),A=new Uint32Array(4),P=new Int32Array(4),L=new oe;let z=null,D=null;const F=[],R=[];let I=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Di,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const W=this;let O=!1,j=null;this._outputColorSpace=ti;let re=0,ae=0,X=null,Z=-1,q=null;const G=new Jt,J=new Jt;let ie=null;const U=new At(0);let K=0,Le=t.width,De=t.height,we=1,se=null,_e=null;const de=new Jt(0,0,Le,De),Ie=new Jt(0,0,Le,De);let je=!1;const $e=new L_;let Ut=!1,ct=!1;const Et=new rn,Dt=new oe,ft=new Jt,Yt={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let Ft=!1;function hn(){return X===null?we:1}let H=r;function Ot(C,Y){return t.getContext(C,Y)}try{const C={alpha:!0,depth:a,stencil:l,antialias:m,premultipliedAlpha:g,preserveDrawingBuffer:_,powerPreference:M,failIfMajorPerformanceCaveat:u};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${pd}`),t.addEventListener("webglcontextlost",ge,!1),t.addEventListener("webglcontextrestored",We,!1),t.addEventListener("webglcontextcreationerror",st,!1),H===null){const Y="webgl2";if(H=Ot(Y,C),H===null)throw Ot(Y)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(C){throw Mt("WebGLRenderer: "+C.message),C}let dt,Ct,Ne,zt,b,T,$,he,me,ye,Pe,ce,pe,Fe,Be,Ae,Me,et,rt,pt,k,Te,fe;function Oe(){dt=new sM(H),dt.init(),k=new $E(H,dt),Ct=new Zy(H,dt,e,k),Ne=new jE(H,dt),Ct.reversedDepthBuffer&&f&&Ne.buffers.depth.setReversed(!0),zt=new lM(H),b=new NE,T=new KE(H,dt,Ne,b,Ct,k,zt),$=new rM(W),he=new dx(H),Te=new Ky(H,he),me=new oM(H,he,zt,Te),ye=new cM(H,me,he,Te,zt),et=new uM(H,Ct,T),Be=new Qy(b),Pe=new IE(W,$,dt,Ct,Te,Be),ce=new nT(W,b),pe=new FE,Fe=new VE(dt),Me=new jy(W,$,Ne,ye,y,g),Ae=new qE(W,ye,Ct),fe=new iT(H,zt,Ct,Ne),rt=new $y(H,dt,zt),pt=new aM(H,dt,zt),zt.programs=Pe.programs,W.capabilities=Ct,W.extensions=dt,W.properties=b,W.renderLists=pe,W.shadowMap=Ae,W.state=Ne,W.info=zt}Oe(),E!==ii&&(I=new dM(E,t.width,t.height,a,l));const Ce=new eT(W,H);this.xr=Ce,this.getContext=function(){return H},this.getContextAttributes=function(){return H.getContextAttributes()},this.forceContextLoss=function(){const C=dt.get("WEBGL_lose_context");C&&C.loseContext()},this.forceContextRestore=function(){const C=dt.get("WEBGL_lose_context");C&&C.restoreContext()},this.getPixelRatio=function(){return we},this.setPixelRatio=function(C){C!==void 0&&(we=C,this.setSize(Le,De,!1))},this.getSize=function(C){return C.set(Le,De)},this.setSize=function(C,Y,le=!0){if(Ce.isPresenting){tt("WebGLRenderer: Can't change size while VR device is presenting.");return}Le=C,De=Y,t.width=Math.floor(C*we),t.height=Math.floor(Y*we),le===!0&&(t.style.width=C+"px",t.style.height=Y+"px"),I!==null&&I.setSize(t.width,t.height),this.setViewport(0,0,C,Y)},this.getDrawingBufferSize=function(C){return C.set(Le*we,De*we).floor()},this.setDrawingBufferSize=function(C,Y,le){Le=C,De=Y,we=le,t.width=Math.floor(C*le),t.height=Math.floor(Y*le),this.setViewport(0,0,C,Y)},this.setEffects=function(C){if(E===ii){Mt("THREE.WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(C){for(let Y=0;Y<C.length;Y++)if(C[Y].isOutputPass===!0){tt("THREE.WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}I.setEffects(C||[])},this.getCurrentViewport=function(C){return C.copy(G)},this.getViewport=function(C){return C.copy(de)},this.setViewport=function(C,Y,le,te){C.isVector4?de.set(C.x,C.y,C.z,C.w):de.set(C,Y,le,te),Ne.viewport(G.copy(de).multiplyScalar(we).round())},this.getScissor=function(C){return C.copy(Ie)},this.setScissor=function(C,Y,le,te){C.isVector4?Ie.set(C.x,C.y,C.z,C.w):Ie.set(C,Y,le,te),Ne.scissor(J.copy(Ie).multiplyScalar(we).round())},this.getScissorTest=function(){return je},this.setScissorTest=function(C){Ne.setScissorTest(je=C)},this.setOpaqueSort=function(C){se=C},this.setTransparentSort=function(C){_e=C},this.getClearColor=function(C){return C.copy(Me.getClearColor())},this.setClearColor=function(){Me.setClearColor(...arguments)},this.getClearAlpha=function(){return Me.getClearAlpha()},this.setClearAlpha=function(){Me.setClearAlpha(...arguments)},this.clear=function(C=!0,Y=!0,le=!0){let te=0;if(C){let ee=!1;if(X!==null){const be=X.texture.format;ee=S.has(be)}if(ee){const be=X.texture.type,He=v.has(be),Re=Me.getClearColor(),Xe=Me.getClearAlpha(),Ze=Re.r,ot=Re.g,at=Re.b;He?(A[0]=Ze,A[1]=ot,A[2]=at,A[3]=Xe,H.clearBufferuiv(H.COLOR,0,A)):(P[0]=Ze,P[1]=ot,P[2]=at,P[3]=Xe,H.clearBufferiv(H.COLOR,0,P))}else te|=H.COLOR_BUFFER_BIT}Y&&(te|=H.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),le&&(te|=H.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),te!==0&&H.clear(te)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(C){C.setRenderer(this),j=C},this.dispose=function(){t.removeEventListener("webglcontextlost",ge,!1),t.removeEventListener("webglcontextrestored",We,!1),t.removeEventListener("webglcontextcreationerror",st,!1),Me.dispose(),pe.dispose(),Fe.dispose(),b.dispose(),$.dispose(),ye.dispose(),Te.dispose(),fe.dispose(),Pe.dispose(),Ce.dispose(),Ce.removeEventListener("sessionstart",Ir),Ce.removeEventListener("sessionend",cs),Fi.stop()};function ge(C){C.preventDefault(),fm("WebGLRenderer: Context Lost."),O=!0}function We(){fm("WebGLRenderer: Context Restored."),O=!1;const C=zt.autoReset,Y=Ae.enabled,le=Ae.autoUpdate,te=Ae.needsUpdate,ee=Ae.type;Oe(),zt.autoReset=C,Ae.enabled=Y,Ae.autoUpdate=le,Ae.needsUpdate=te,Ae.type=ee}function st(C){Mt("WebGLRenderer: A WebGL context could not be created. Reason: ",C.statusMessage)}function Nt(C){const Y=C.target;Y.removeEventListener("dispose",Nt),Tt(Y)}function Tt(C){wn(C),b.remove(C)}function wn(C){const Y=b.get(C).programs;Y!==void 0&&(Y.forEach(function(le){Pe.releaseProgram(le)}),C.isShaderMaterial&&Pe.releaseShaderCache(C))}this.renderBufferDirect=function(C,Y,le,te,ee,be){Y===null&&(Y=Yt);const He=ee.isMesh&&ee.matrixWorld.determinant()<0,Re=fa(C,Y,le,te,ee);Ne.setMaterial(te,He);let Xe=le.index,Ze=1;if(te.wireframe===!0){if(Xe=me.getWireframeAttribute(le),Xe===void 0)return;Ze=2}const ot=le.drawRange,at=le.attributes.position;let qe=ot.start*Ze,St=(ot.start+ot.count)*Ze;be!==null&&(qe=Math.max(qe,be.start*Ze),St=Math.min(St,(be.start+be.count)*Ze)),Xe!==null?(qe=Math.max(qe,0),St=Math.min(St,Xe.count)):at!=null&&(qe=Math.max(qe,0),St=Math.min(St,at.count));const Bt=St-qe;if(Bt<0||Bt===1/0)return;Te.setup(ee,te,Re,le,Xe);let Wt,bt=rt;if(Xe!==null&&(Wt=he.get(Xe),bt=pt,bt.setIndex(Wt)),ee.isMesh)te.wireframe===!0?(Ne.setLineWidth(te.wireframeLinewidth*hn()),bt.setMode(H.LINES)):bt.setMode(H.TRIANGLES);else if(ee.isLine){let en=te.linewidth;en===void 0&&(en=1),Ne.setLineWidth(en*hn()),ee.isLineSegments?bt.setMode(H.LINES):ee.isLineLoop?bt.setMode(H.LINE_LOOP):bt.setMode(H.LINE_STRIP)}else ee.isPoints?bt.setMode(H.POINTS):ee.isSprite&&bt.setMode(H.TRIANGLES);if(ee.isBatchedMesh)if(dt.get("WEBGL_multi_draw"))bt.renderMultiDraw(ee._multiDrawStarts,ee._multiDrawCounts,ee._multiDrawCount);else{const en=ee._multiDrawStarts,ke=ee._multiDrawCounts,pn=ee._multiDrawCount,mt=Xe?he.get(Xe).bytesPerElement:1,Ln=b.get(te).currentProgram.getUniforms();for(let Dn=0;Dn<pn;Dn++)Ln.setValue(H,"_gl_DrawID",Dn),bt.render(en[Dn]/mt,ke[Dn])}else if(ee.isInstancedMesh)bt.renderInstances(qe,Bt,ee.count);else if(le.isInstancedBufferGeometry){const en=le._maxInstanceCount!==void 0?le._maxInstanceCount:1/0,ke=Math.min(le.instanceCount,en);bt.renderInstances(qe,Bt,ke)}else bt.render(qe,Bt)};function qn(C,Y,le){C.transparent===!0&&C.side===Zi&&C.forceSinglePass===!1?(C.side=kn,C.needsUpdate=!0,fs(C,Y,le),C.side=Dr,C.needsUpdate=!0,fs(C,Y,le),C.side=Zi):fs(C,Y,le)}this.compile=function(C,Y,le=null){le===null&&(le=C),D=Fe.get(le),D.init(Y),R.push(D),le.traverseVisible(function(ee){ee.isLight&&ee.layers.test(Y.layers)&&(D.pushLight(ee),ee.castShadow&&D.pushShadow(ee))}),C!==le&&C.traverseVisible(function(ee){ee.isLight&&ee.layers.test(Y.layers)&&(D.pushLight(ee),ee.castShadow&&D.pushShadow(ee))}),D.setupLights();const te=new Set;return C.traverse(function(ee){if(!(ee.isMesh||ee.isPoints||ee.isLine||ee.isSprite))return;const be=ee.material;if(be)if(Array.isArray(be))for(let He=0;He<be.length;He++){const Re=be[He];qn(Re,le,ee),te.add(Re)}else qn(be,le,ee),te.add(be)}),D=R.pop(),te},this.compileAsync=function(C,Y,le=null){const te=this.compile(C,Y,le);return new Promise(ee=>{function be(){if(te.forEach(function(He){b.get(He).currentProgram.isReady()&&te.delete(He)}),te.size===0){ee(C);return}setTimeout(be,10)}dt.get("KHR_parallel_shader_compile")!==null?be():setTimeout(be,10)})};let Ui=null;function us(C){Ui&&Ui(C)}function Ir(){Fi.stop()}function cs(){Fi.start()}const Fi=new B_;Fi.setAnimationLoop(us),typeof self<"u"&&Fi.setContext(self),this.setAnimationLoop=function(C){Ui=C,Ce.setAnimationLoop(C),C===null?Fi.stop():Fi.start()},Ce.addEventListener("sessionstart",Ir),Ce.addEventListener("sessionend",cs),this.render=function(C,Y){if(Y!==void 0&&Y.isCamera!==!0){Mt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(O===!0)return;j!==null&&j.renderStart(C,Y);const le=Ce.enabled===!0&&Ce.isPresenting===!0,te=I!==null&&(X===null||le)&&I.begin(W,X);if(C.matrixWorldAutoUpdate===!0&&C.updateMatrixWorld(),Y.parent===null&&Y.matrixWorldAutoUpdate===!0&&Y.updateMatrixWorld(),Ce.enabled===!0&&Ce.isPresenting===!0&&(I===null||I.isCompositing()===!1)&&(Ce.cameraAutoUpdate===!0&&Ce.updateCamera(Y),Y=Ce.getCamera()),C.isScene===!0&&C.onBeforeRender(W,C,Y,X),D=Fe.get(C,R.length),D.init(Y),D.state.textureUnits=T.getTextureUnits(),R.push(D),Et.multiplyMatrices(Y.projectionMatrix,Y.matrixWorldInverse),$e.setFromProjectionMatrix(Et,Li,Y.reversedDepth),ct=this.localClippingEnabled,Ut=Be.init(this.clippingPlanes,ct),z=pe.get(C,F.length),z.init(),F.push(z),Ce.enabled===!0&&Ce.isPresenting===!0){const He=W.xr.getDepthSensingMesh();He!==null&&so(He,Y,-1/0,W.sortObjects)}so(C,Y,0,W.sortObjects),z.finish(),W.sortObjects===!0&&z.sort(se,_e),Ft=Ce.enabled===!1||Ce.isPresenting===!1||Ce.hasDepthSensing()===!1,Ft&&Me.addToRenderList(z,C),this.info.render.frame++,Ut===!0&&Be.beginShadows();const ee=D.state.shadowsArray;if(Ae.render(ee,C,Y),Ut===!0&&Be.endShadows(),this.info.autoReset===!0&&this.info.reset(),(te&&I.hasRenderPass())===!1){const He=z.opaque,Re=z.transmissive;if(D.setupLights(),Y.isArrayCamera){const Xe=Y.cameras;if(Re.length>0)for(let Ze=0,ot=Xe.length;Ze<ot;Ze++){const at=Xe[Ze];Si(He,Re,C,at)}Ft&&Me.render(C);for(let Ze=0,ot=Xe.length;Ze<ot;Ze++){const at=Xe[Ze];ua(z,C,at,at.viewport)}}else Re.length>0&&Si(He,Re,C,Y),Ft&&Me.render(C),ua(z,C,Y)}X!==null&&ae===0&&(T.updateMultisampleRenderTarget(X),T.updateRenderTargetMipmap(X)),te&&I.end(W),C.isScene===!0&&C.onAfterRender(W,C,Y),Te.resetDefaultState(),Z=-1,q=null,R.pop(),R.length>0?(D=R[R.length-1],T.setTextureUnits(D.state.textureUnits),Ut===!0&&Be.setGlobalState(W.clippingPlanes,D.state.camera)):D=null,F.pop(),F.length>0?z=F[F.length-1]:z=null,j!==null&&j.renderEnd()};function so(C,Y,le,te){if(C.visible===!1)return;if(C.layers.test(Y.layers)){if(C.isGroup)le=C.renderOrder;else if(C.isLOD)C.autoUpdate===!0&&C.update(Y);else if(C.isLightProbeGrid)D.pushLightProbeGrid(C);else if(C.isLight)D.pushLight(C),C.castShadow&&D.pushShadow(C);else if(C.isSprite){if(!C.frustumCulled||$e.intersectsSprite(C)){te&&ft.setFromMatrixPosition(C.matrixWorld).applyMatrix4(Et);const He=ye.update(C),Re=C.material;Re.visible&&z.push(C,He,Re,le,ft.z,null)}}else if((C.isMesh||C.isLine||C.isPoints)&&(!C.frustumCulled||$e.intersectsObject(C))){const He=ye.update(C),Re=C.material;if(te&&(C.boundingSphere!==void 0?(C.boundingSphere===null&&C.computeBoundingSphere(),ft.copy(C.boundingSphere.center)):(He.boundingSphere===null&&He.computeBoundingSphere(),ft.copy(He.boundingSphere.center)),ft.applyMatrix4(C.matrixWorld).applyMatrix4(Et)),Array.isArray(Re)){const Xe=He.groups;for(let Ze=0,ot=Xe.length;Ze<ot;Ze++){const at=Xe[Ze],qe=Re[at.materialIndex];qe&&qe.visible&&z.push(C,He,qe,le,ft.z,at)}}else Re.visible&&z.push(C,He,Re,le,ft.z,null)}}const be=C.children;for(let He=0,Re=be.length;He<Re;He++)so(be[He],Y,le,te)}function ua(C,Y,le,te){const{opaque:ee,transmissive:be,transparent:He}=C;D.setupLightsView(le),Ut===!0&&Be.setGlobalState(W.clippingPlanes,le),te&&Ne.viewport(G.copy(te)),ee.length>0&&Nr(ee,Y,le),be.length>0&&Nr(be,Y,le),He.length>0&&Nr(He,Y,le),Ne.buffers.depth.setTest(!0),Ne.buffers.depth.setMask(!0),Ne.buffers.color.setMask(!0),Ne.setPolygonOffset(!1)}function Si(C,Y,le,te){if((le.isScene===!0?le.overrideMaterial:null)!==null)return;if(D.state.transmissionRenderTarget[te.id]===void 0){const qe=dt.has("EXT_color_buffer_half_float")||dt.has("EXT_color_buffer_float");D.state.transmissionRenderTarget[te.id]=new Ii(1,1,{generateMipmaps:!0,type:qe?nr:ii,minFilter:ns,samples:Math.max(4,Ct.samples),stencilBuffer:l,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:xt.workingColorSpace})}const be=D.state.transmissionRenderTarget[te.id],He=te.viewport||G;be.setSize(He.z*W.transmissionResolutionScale,He.w*W.transmissionResolutionScale);const Re=W.getRenderTarget(),Xe=W.getActiveCubeFace(),Ze=W.getActiveMipmapLevel();W.setRenderTarget(be),W.getClearColor(U),K=W.getClearAlpha(),K<1&&W.setClearColor(16777215,.5),W.clear(),Ft&&Me.render(le);const ot=W.toneMapping;W.toneMapping=Di;const at=te.viewport;if(te.viewport!==void 0&&(te.viewport=void 0),D.setupLightsView(te),Ut===!0&&Be.setGlobalState(W.clippingPlanes,te),Nr(C,le,te),T.updateMultisampleRenderTarget(be),T.updateRenderTargetMipmap(be),dt.has("WEBGL_multisampled_render_to_texture")===!1){let qe=!1;for(let St=0,Bt=Y.length;St<Bt;St++){const Wt=Y[St],{object:bt,geometry:en,material:ke,group:pn}=Wt;if(ke.side===Zi&&bt.layers.test(te.layers)){const mt=ke.side;ke.side=kn,ke.needsUpdate=!0,oo(bt,le,te,en,ke,pn),ke.side=mt,ke.needsUpdate=!0,qe=!0}}qe===!0&&(T.updateMultisampleRenderTarget(be),T.updateRenderTargetMipmap(be))}W.setRenderTarget(Re,Xe,Ze),W.setClearColor(U,K),at!==void 0&&(te.viewport=at),W.toneMapping=ot}function Nr(C,Y,le){const te=Y.isScene===!0?Y.overrideMaterial:null;for(let ee=0,be=C.length;ee<be;ee++){const He=C[ee],{object:Re,geometry:Xe,group:Ze}=He;let ot=He.material;ot.allowOverride===!0&&te!==null&&(ot=te),Re.layers.test(le.layers)&&oo(Re,Y,le,Xe,ot,Ze)}}function oo(C,Y,le,te,ee,be){C.onBeforeRender(W,Y,le,te,ee,be),C.modelViewMatrix.multiplyMatrices(le.matrixWorldInverse,C.matrixWorld),C.normalMatrix.getNormalMatrix(C.modelViewMatrix),ee.onBeforeRender(W,Y,le,te,C,be),ee.transparent===!0&&ee.side===Zi&&ee.forceSinglePass===!1?(ee.side=kn,ee.needsUpdate=!0,W.renderBufferDirect(le,Y,te,ee,C,be),ee.side=Dr,ee.needsUpdate=!0,W.renderBufferDirect(le,Y,te,ee,C,be),ee.side=Zi):W.renderBufferDirect(le,Y,te,ee,C,be),C.onAfterRender(W,Y,le,te,ee,be)}function fs(C,Y,le){Y.isScene!==!0&&(Y=Yt);const te=b.get(C),ee=D.state.lights,be=D.state.shadowsArray,He=ee.state.version,Re=Pe.getParameters(C,ee.state,be,Y,le,D.state.lightProbeGridArray),Xe=Pe.getProgramCacheKey(Re);let Ze=te.programs;te.environment=C.isMeshStandardMaterial||C.isMeshLambertMaterial||C.isMeshPhongMaterial?Y.environment:null,te.fog=Y.fog;const ot=C.isMeshStandardMaterial||C.isMeshLambertMaterial&&!C.envMap||C.isMeshPhongMaterial&&!C.envMap;te.envMap=$.get(C.envMap||te.environment,ot),te.envMapRotation=te.environment!==null&&C.envMap===null?Y.environmentRotation:C.envMapRotation,Ze===void 0&&(C.addEventListener("dispose",Nt),Ze=new Map,te.programs=Ze);let at=Ze.get(Xe);if(at!==void 0){if(te.currentProgram===at&&te.lightsStateVersion===He)return lo(C,Re),at}else Re.uniforms=Pe.getUniforms(C),j!==null&&C.isNodeMaterial&&j.build(C,le,Re),C.onBeforeCompile(Re,W),at=Pe.acquireProgram(Re,Xe),Ze.set(Xe,at),te.uniforms=Re.uniforms;const qe=te.uniforms;return(!C.isShaderMaterial&&!C.isRawShaderMaterial||C.clipping===!0)&&(qe.clippingPlanes=Be.uniform),lo(C,Re),te.needsLights=ou(C),te.lightsStateVersion=He,te.needsLights&&(qe.ambientLightColor.value=ee.state.ambient,qe.lightProbe.value=ee.state.probe,qe.directionalLights.value=ee.state.directional,qe.directionalLightShadows.value=ee.state.directionalShadow,qe.spotLights.value=ee.state.spot,qe.spotLightShadows.value=ee.state.spotShadow,qe.rectAreaLights.value=ee.state.rectArea,qe.ltc_1.value=ee.state.rectAreaLTC1,qe.ltc_2.value=ee.state.rectAreaLTC2,qe.pointLights.value=ee.state.point,qe.pointLightShadows.value=ee.state.pointShadow,qe.hemisphereLights.value=ee.state.hemi,qe.directionalShadowMatrix.value=ee.state.directionalShadowMatrix,qe.spotLightMatrix.value=ee.state.spotLightMatrix,qe.spotLightMap.value=ee.state.spotLightMap,qe.pointShadowMatrix.value=ee.state.pointShadowMatrix),te.lightProbeGrid=D.state.lightProbeGridArray.length>0,te.currentProgram=at,te.uniformsList=null,at}function ao(C){if(C.uniformsList===null){const Y=C.currentProgram.getUniforms();C.uniformsList=Wl.seqWithValue(Y.seq,C.uniforms)}return C.uniformsList}function lo(C,Y){const le=b.get(C);le.outputColorSpace=Y.outputColorSpace,le.batching=Y.batching,le.batchingColor=Y.batchingColor,le.instancing=Y.instancing,le.instancingColor=Y.instancingColor,le.instancingMorph=Y.instancingMorph,le.skinning=Y.skinning,le.morphTargets=Y.morphTargets,le.morphNormals=Y.morphNormals,le.morphColors=Y.morphColors,le.morphTargetsCount=Y.morphTargetsCount,le.numClippingPlanes=Y.numClippingPlanes,le.numIntersection=Y.numClipIntersection,le.vertexAlphas=Y.vertexAlphas,le.vertexTangents=Y.vertexTangents,le.toneMapping=Y.toneMapping}function ca(C,Y){if(C.length===0)return null;if(C.length===1)return C[0].texture!==null?C[0]:null;L.setFromMatrixPosition(Y.matrixWorld);for(let le=0,te=C.length;le<te;le++){const ee=C[le];if(ee.texture!==null&&ee.boundingBox.containsPoint(L))return ee}return null}function fa(C,Y,le,te,ee){Y.isScene!==!0&&(Y=Yt),T.resetTextureUnits();const be=Y.fog,He=te.isMeshStandardMaterial||te.isMeshLambertMaterial||te.isMeshPhongMaterial?Y.environment:null,Re=X===null?W.outputColorSpace:X.isXRRenderTarget===!0?X.texture.colorSpace:xt.workingColorSpace,Xe=te.isMeshStandardMaterial||te.isMeshLambertMaterial&&!te.envMap||te.isMeshPhongMaterial&&!te.envMap,Ze=$.get(te.envMap||He,Xe),ot=te.vertexColors===!0&&!!le.attributes.color&&le.attributes.color.itemSize===4,at=!!le.attributes.tangent&&(!!te.normalMap||te.anisotropy>0),qe=!!le.morphAttributes.position,St=!!le.morphAttributes.normal,Bt=!!le.morphAttributes.color;let Wt=Di;te.toneMapped&&(X===null||X.isXRRenderTarget===!0)&&(Wt=W.toneMapping);const bt=le.morphAttributes.position||le.morphAttributes.normal||le.morphAttributes.color,en=bt!==void 0?bt.length:0,ke=b.get(te),pn=D.state.lights;if(Ut===!0&&(ct===!0||C!==q)){const Pt=C===q&&te.id===Z;Be.setState(te,C,Pt)}let mt=!1;te.version===ke.__version?(ke.needsLights&&ke.lightsStateVersion!==pn.state.version||ke.outputColorSpace!==Re||ee.isBatchedMesh&&ke.batching===!1||!ee.isBatchedMesh&&ke.batching===!0||ee.isBatchedMesh&&ke.batchingColor===!0&&ee.colorTexture===null||ee.isBatchedMesh&&ke.batchingColor===!1&&ee.colorTexture!==null||ee.isInstancedMesh&&ke.instancing===!1||!ee.isInstancedMesh&&ke.instancing===!0||ee.isSkinnedMesh&&ke.skinning===!1||!ee.isSkinnedMesh&&ke.skinning===!0||ee.isInstancedMesh&&ke.instancingColor===!0&&ee.instanceColor===null||ee.isInstancedMesh&&ke.instancingColor===!1&&ee.instanceColor!==null||ee.isInstancedMesh&&ke.instancingMorph===!0&&ee.morphTexture===null||ee.isInstancedMesh&&ke.instancingMorph===!1&&ee.morphTexture!==null||ke.envMap!==Ze||te.fog===!0&&ke.fog!==be||ke.numClippingPlanes!==void 0&&(ke.numClippingPlanes!==Be.numPlanes||ke.numIntersection!==Be.numIntersection)||ke.vertexAlphas!==ot||ke.vertexTangents!==at||ke.morphTargets!==qe||ke.morphNormals!==St||ke.morphColors!==Bt||ke.toneMapping!==Wt||ke.morphTargetsCount!==en||!!ke.lightProbeGrid!=D.state.lightProbeGridArray.length>0)&&(mt=!0):(mt=!0,ke.__version=te.version);let Ln=ke.currentProgram;mt===!0&&(Ln=fs(te,Y,ee),j&&te.isNodeMaterial&&j.onUpdateProgram(te,Ln,ke));let Dn=!1,_t=!1,Oi=!1;const Rt=Ln.getUniforms(),Ht=ke.uniforms;if(Ne.useProgram(Ln.program)&&(Dn=!0,_t=!0,Oi=!0),te.id!==Z&&(Z=te.id,_t=!0),ke.needsLights){const Pt=ca(D.state.lightProbeGridArray,ee);ke.lightProbeGrid!==Pt&&(ke.lightProbeGrid=Pt,_t=!0)}if(Dn||q!==C){Ne.buffers.depth.getReversed()&&C.reversedDepth!==!0&&(C._reversedDepth=!0,C.updateProjectionMatrix()),Rt.setValue(H,"projectionMatrix",C.projectionMatrix),Rt.setValue(H,"viewMatrix",C.matrixWorldInverse);const oi=Rt.map.cameraPosition;oi!==void 0&&oi.setValue(H,Dt.setFromMatrixPosition(C.matrixWorld)),Ct.logarithmicDepthBuffer&&Rt.setValue(H,"logDepthBufFC",2/(Math.log(C.far+1)/Math.LN2)),(te.isMeshPhongMaterial||te.isMeshToonMaterial||te.isMeshLambertMaterial||te.isMeshBasicMaterial||te.isMeshStandardMaterial||te.isShaderMaterial)&&Rt.setValue(H,"isOrthographic",C.isOrthographicCamera===!0),q!==C&&(q=C,_t=!0,Oi=!0)}if(ke.needsLights&&(pn.state.directionalShadowMap.length>0&&Rt.setValue(H,"directionalShadowMap",pn.state.directionalShadowMap,T),pn.state.spotShadowMap.length>0&&Rt.setValue(H,"spotShadowMap",pn.state.spotShadowMap,T),pn.state.pointShadowMap.length>0&&Rt.setValue(H,"pointShadowMap",pn.state.pointShadowMap,T)),ee.isSkinnedMesh){Rt.setOptional(H,ee,"bindMatrix"),Rt.setOptional(H,ee,"bindMatrixInverse");const Pt=ee.skeleton;Pt&&(Pt.boneTexture===null&&Pt.computeBoneTexture(),Rt.setValue(H,"boneTexture",Pt.boneTexture,T))}ee.isBatchedMesh&&(Rt.setOptional(H,ee,"batchingTexture"),Rt.setValue(H,"batchingTexture",ee._matricesTexture,T),Rt.setOptional(H,ee,"batchingIdTexture"),Rt.setValue(H,"batchingIdTexture",ee._indirectTexture,T),Rt.setOptional(H,ee,"batchingColorTexture"),ee._colorsTexture!==null&&Rt.setValue(H,"batchingColorTexture",ee._colorsTexture,T));const si=le.morphAttributes;if((si.position!==void 0||si.normal!==void 0||si.color!==void 0)&&et.update(ee,le,Ln),(_t||ke.receiveShadow!==ee.receiveShadow)&&(ke.receiveShadow=ee.receiveShadow,Rt.setValue(H,"receiveShadow",ee.receiveShadow)),(te.isMeshStandardMaterial||te.isMeshLambertMaterial||te.isMeshPhongMaterial)&&te.envMap===null&&Y.environment!==null&&(Ht.envMapIntensity.value=Y.environmentIntensity),Ht.dfgLUT!==void 0&&(Ht.dfgLUT.value=sT()),_t){if(Rt.setValue(H,"toneMappingExposure",W.toneMappingExposure),ke.needsLights&&su(Ht,Oi),be&&te.fog===!0&&ce.refreshFogUniforms(Ht,be),ce.refreshMaterialUniforms(Ht,te,we,De,D.state.transmissionRenderTarget[C.id]),ke.needsLights&&ke.lightProbeGrid){const Pt=ke.lightProbeGrid;Ht.probesSH.value=Pt.texture,Ht.probesMin.value.copy(Pt.boundingBox.min),Ht.probesMax.value.copy(Pt.boundingBox.max),Ht.probesResolution.value.copy(Pt.resolution)}Wl.upload(H,ao(ke),Ht,T)}if(te.isShaderMaterial&&te.uniformsNeedUpdate===!0&&(Wl.upload(H,ao(ke),Ht,T),te.uniformsNeedUpdate=!1),te.isSpriteMaterial&&Rt.setValue(H,"center",ee.center),Rt.setValue(H,"modelViewMatrix",ee.modelViewMatrix),Rt.setValue(H,"normalMatrix",ee.normalMatrix),Rt.setValue(H,"modelMatrix",ee.matrixWorld),te.uniformsGroups!==void 0){const Pt=te.uniformsGroups;for(let oi=0,yi=Pt.length;oi<yi;oi++){const Ur=Pt[oi];fe.update(Ur,Ln),fe.bind(Ur,Ln)}}return Ln}function su(C,Y){C.ambientLightColor.needsUpdate=Y,C.lightProbe.needsUpdate=Y,C.directionalLights.needsUpdate=Y,C.directionalLightShadows.needsUpdate=Y,C.pointLights.needsUpdate=Y,C.pointLightShadows.needsUpdate=Y,C.spotLights.needsUpdate=Y,C.spotLightShadows.needsUpdate=Y,C.rectAreaLights.needsUpdate=Y,C.hemisphereLights.needsUpdate=Y}function ou(C){return C.isMeshLambertMaterial||C.isMeshToonMaterial||C.isMeshPhongMaterial||C.isMeshStandardMaterial||C.isShadowMaterial||C.isShaderMaterial&&C.lights===!0}this.getActiveCubeFace=function(){return re},this.getActiveMipmapLevel=function(){return ae},this.getRenderTarget=function(){return X},this.setRenderTargetTextures=function(C,Y,le){const te=b.get(C);te.__autoAllocateDepthBuffer=C.resolveDepthBuffer===!1,te.__autoAllocateDepthBuffer===!1&&(te.__useRenderToTexture=!1),b.get(C.texture).__webglTexture=Y,b.get(C.depthTexture).__webglTexture=te.__autoAllocateDepthBuffer?void 0:le,te.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(C,Y){const le=b.get(C);le.__webglFramebuffer=Y,le.__useDefaultFramebuffer=Y===void 0};const qt=H.createFramebuffer();this.setRenderTarget=function(C,Y=0,le=0){X=C,re=Y,ae=le;let te=null,ee=!1,be=!1;if(C){const Re=b.get(C);if(Re.__useDefaultFramebuffer!==void 0){Ne.bindFramebuffer(H.FRAMEBUFFER,Re.__webglFramebuffer),G.copy(C.viewport),J.copy(C.scissor),ie=C.scissorTest,Ne.viewport(G),Ne.scissor(J),Ne.setScissorTest(ie),Z=-1;return}else if(Re.__webglFramebuffer===void 0)T.setupRenderTarget(C);else if(Re.__hasExternalTextures)T.rebindTextures(C,b.get(C.texture).__webglTexture,b.get(C.depthTexture).__webglTexture);else if(C.depthBuffer){const ot=C.depthTexture;if(Re.__boundDepthTexture!==ot){if(ot!==null&&b.has(ot)&&(C.width!==ot.image.width||C.height!==ot.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");T.setupDepthRenderbuffer(C)}}const Xe=C.texture;(Xe.isData3DTexture||Xe.isDataArrayTexture||Xe.isCompressedArrayTexture)&&(be=!0);const Ze=b.get(C).__webglFramebuffer;C.isWebGLCubeRenderTarget?(Array.isArray(Ze[Y])?te=Ze[Y][le]:te=Ze[Y],ee=!0):C.samples>0&&T.useMultisampledRTT(C)===!1?te=b.get(C).__webglMultisampledFramebuffer:Array.isArray(Ze)?te=Ze[le]:te=Ze,G.copy(C.viewport),J.copy(C.scissor),ie=C.scissorTest}else G.copy(de).multiplyScalar(we).floor(),J.copy(Ie).multiplyScalar(we).floor(),ie=je;if(le!==0&&(te=qt),Ne.bindFramebuffer(H.FRAMEBUFFER,te)&&Ne.drawBuffers(C,te),Ne.viewport(G),Ne.scissor(J),Ne.setScissorTest(ie),ee){const Re=b.get(C.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_CUBE_MAP_POSITIVE_X+Y,Re.__webglTexture,le)}else if(be){const Re=Y;for(let Xe=0;Xe<C.textures.length;Xe++){const Ze=b.get(C.textures[Xe]);H.framebufferTextureLayer(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0+Xe,Ze.__webglTexture,le,Re)}}else if(C!==null&&le!==0){const Re=b.get(C.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,Re.__webglTexture,le)}Z=-1},this.readRenderTargetPixels=function(C,Y,le,te,ee,be,He,Re=0){if(!(C&&C.isWebGLRenderTarget)){Mt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Xe=b.get(C).__webglFramebuffer;if(C.isWebGLCubeRenderTarget&&He!==void 0&&(Xe=Xe[He]),Xe){Ne.bindFramebuffer(H.FRAMEBUFFER,Xe);try{const Ze=C.textures[Re],ot=Ze.format,at=Ze.type;if(C.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+Re),!Ct.textureFormatReadable(ot)){Mt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!Ct.textureTypeReadable(at)){Mt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Y>=0&&Y<=C.width-te&&le>=0&&le<=C.height-ee&&H.readPixels(Y,le,te,ee,k.convert(ot),k.convert(at),be)}finally{const Ze=X!==null?b.get(X).__webglFramebuffer:null;Ne.bindFramebuffer(H.FRAMEBUFFER,Ze)}}},this.readRenderTargetPixelsAsync=async function(C,Y,le,te,ee,be,He,Re=0){if(!(C&&C.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Xe=b.get(C).__webglFramebuffer;if(C.isWebGLCubeRenderTarget&&He!==void 0&&(Xe=Xe[He]),Xe)if(Y>=0&&Y<=C.width-te&&le>=0&&le<=C.height-ee){Ne.bindFramebuffer(H.FRAMEBUFFER,Xe);const Ze=C.textures[Re],ot=Ze.format,at=Ze.type;if(C.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+Re),!Ct.textureFormatReadable(ot))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!Ct.textureTypeReadable(at))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const qe=H.createBuffer();H.bindBuffer(H.PIXEL_PACK_BUFFER,qe),H.bufferData(H.PIXEL_PACK_BUFFER,be.byteLength,H.STREAM_READ),H.readPixels(Y,le,te,ee,k.convert(ot),k.convert(at),0);const St=X!==null?b.get(X).__webglFramebuffer:null;Ne.bindFramebuffer(H.FRAMEBUFFER,St);const Bt=H.fenceSync(H.SYNC_GPU_COMMANDS_COMPLETE,0);return H.flush(),await hv(H,Bt,4),H.bindBuffer(H.PIXEL_PACK_BUFFER,qe),H.getBufferSubData(H.PIXEL_PACK_BUFFER,0,be),H.deleteBuffer(qe),H.deleteSync(Bt),be}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(C,Y=null,le=0){const te=Math.pow(2,-le),ee=Math.floor(C.image.width*te),be=Math.floor(C.image.height*te),He=Y!==null?Y.x:0,Re=Y!==null?Y.y:0;T.setTexture2D(C,0),H.copyTexSubImage2D(H.TEXTURE_2D,le,0,0,He,Re,ee,be),Ne.unbindTexture()};const au=H.createFramebuffer(),uo=H.createFramebuffer();this.copyTextureToTexture=function(C,Y,le=null,te=null,ee=0,be=0){let He,Re,Xe,Ze,ot,at,qe,St,Bt;const Wt=C.isCompressedTexture?C.mipmaps[be]:C.image;if(le!==null)He=le.max.x-le.min.x,Re=le.max.y-le.min.y,Xe=le.isBox3?le.max.z-le.min.z:1,Ze=le.min.x,ot=le.min.y,at=le.isBox3?le.min.z:0;else{const Ht=Math.pow(2,-ee);He=Math.floor(Wt.width*Ht),Re=Math.floor(Wt.height*Ht),C.isDataArrayTexture?Xe=Wt.depth:C.isData3DTexture?Xe=Math.floor(Wt.depth*Ht):Xe=1,Ze=0,ot=0,at=0}te!==null?(qe=te.x,St=te.y,Bt=te.z):(qe=0,St=0,Bt=0);const bt=k.convert(Y.format),en=k.convert(Y.type);let ke;Y.isData3DTexture?(T.setTexture3D(Y,0),ke=H.TEXTURE_3D):Y.isDataArrayTexture||Y.isCompressedArrayTexture?(T.setTexture2DArray(Y,0),ke=H.TEXTURE_2D_ARRAY):(T.setTexture2D(Y,0),ke=H.TEXTURE_2D),Ne.activeTexture(H.TEXTURE0),Ne.pixelStorei(H.UNPACK_FLIP_Y_WEBGL,Y.flipY),Ne.pixelStorei(H.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Y.premultiplyAlpha),Ne.pixelStorei(H.UNPACK_ALIGNMENT,Y.unpackAlignment);const pn=Ne.getParameter(H.UNPACK_ROW_LENGTH),mt=Ne.getParameter(H.UNPACK_IMAGE_HEIGHT),Ln=Ne.getParameter(H.UNPACK_SKIP_PIXELS),Dn=Ne.getParameter(H.UNPACK_SKIP_ROWS),_t=Ne.getParameter(H.UNPACK_SKIP_IMAGES);Ne.pixelStorei(H.UNPACK_ROW_LENGTH,Wt.width),Ne.pixelStorei(H.UNPACK_IMAGE_HEIGHT,Wt.height),Ne.pixelStorei(H.UNPACK_SKIP_PIXELS,Ze),Ne.pixelStorei(H.UNPACK_SKIP_ROWS,ot),Ne.pixelStorei(H.UNPACK_SKIP_IMAGES,at);const Oi=C.isDataArrayTexture||C.isData3DTexture,Rt=Y.isDataArrayTexture||Y.isData3DTexture;if(C.isDepthTexture){const Ht=b.get(C),si=b.get(Y),Pt=b.get(Ht.__renderTarget),oi=b.get(si.__renderTarget);Ne.bindFramebuffer(H.READ_FRAMEBUFFER,Pt.__webglFramebuffer),Ne.bindFramebuffer(H.DRAW_FRAMEBUFFER,oi.__webglFramebuffer);for(let yi=0;yi<Xe;yi++)Oi&&(H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,b.get(C).__webglTexture,ee,at+yi),H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,b.get(Y).__webglTexture,be,Bt+yi)),H.blitFramebuffer(Ze,ot,He,Re,qe,St,He,Re,H.DEPTH_BUFFER_BIT,H.NEAREST);Ne.bindFramebuffer(H.READ_FRAMEBUFFER,null),Ne.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else if(ee!==0||C.isRenderTargetTexture||b.has(C)){const Ht=b.get(C),si=b.get(Y);Ne.bindFramebuffer(H.READ_FRAMEBUFFER,au),Ne.bindFramebuffer(H.DRAW_FRAMEBUFFER,uo);for(let Pt=0;Pt<Xe;Pt++)Oi?H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,Ht.__webglTexture,ee,at+Pt):H.framebufferTexture2D(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,Ht.__webglTexture,ee),Rt?H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,si.__webglTexture,be,Bt+Pt):H.framebufferTexture2D(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,si.__webglTexture,be),ee!==0?H.blitFramebuffer(Ze,ot,He,Re,qe,St,He,Re,H.COLOR_BUFFER_BIT,H.NEAREST):Rt?H.copyTexSubImage3D(ke,be,qe,St,Bt+Pt,Ze,ot,He,Re):H.copyTexSubImage2D(ke,be,qe,St,Ze,ot,He,Re);Ne.bindFramebuffer(H.READ_FRAMEBUFFER,null),Ne.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else Rt?C.isDataTexture||C.isData3DTexture?H.texSubImage3D(ke,be,qe,St,Bt,He,Re,Xe,bt,en,Wt.data):Y.isCompressedArrayTexture?H.compressedTexSubImage3D(ke,be,qe,St,Bt,He,Re,Xe,bt,Wt.data):H.texSubImage3D(ke,be,qe,St,Bt,He,Re,Xe,bt,en,Wt):C.isDataTexture?H.texSubImage2D(H.TEXTURE_2D,be,qe,St,He,Re,bt,en,Wt.data):C.isCompressedTexture?H.compressedTexSubImage2D(H.TEXTURE_2D,be,qe,St,Wt.width,Wt.height,bt,Wt.data):H.texSubImage2D(H.TEXTURE_2D,be,qe,St,He,Re,bt,en,Wt);Ne.pixelStorei(H.UNPACK_ROW_LENGTH,pn),Ne.pixelStorei(H.UNPACK_IMAGE_HEIGHT,mt),Ne.pixelStorei(H.UNPACK_SKIP_PIXELS,Ln),Ne.pixelStorei(H.UNPACK_SKIP_ROWS,Dn),Ne.pixelStorei(H.UNPACK_SKIP_IMAGES,_t),be===0&&Y.generateMipmaps&&H.generateMipmap(ke),Ne.unbindTexture()},this.initRenderTarget=function(C){b.get(C).__webglFramebuffer===void 0&&T.setupRenderTarget(C)},this.initTexture=function(C){C.isCubeTexture?T.setTextureCube(C,0):C.isData3DTexture?T.setTexture3D(C,0):C.isDataArrayTexture||C.isCompressedArrayTexture?T.setTexture2DArray(C,0):T.setTexture2D(C,0),Ne.unbindTexture()},this.resetState=function(){re=0,ae=0,X=null,Ne.reset(),Te.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Li}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=xt._getDrawingBufferColorSpace(e),t.unpackColorSpace=xt._getUnpackColorSpace()}}const xf=s=>Math.min(1,Math.max(0,s)),vi=(s,e)=>s+Math.random()*(e-s),aT=s=>{const e=s.replace("#","").trim(),t=e.length===3?e.split("").map(r=>r+r).join(""):e.padEnd(6,"0").slice(0,6);return{r:parseInt(t.slice(0,2),16)/255,g:parseInt(t.slice(2,4),16)/255,b:parseInt(t.slice(4,6),16)/255}},to=(s,e,t)=>({r:s.r*(1-t)+e.r*t,g:s.g*(1-t)+e.g*t,b:s.b*(1-t)+e.b*t}),sa=(s,e,t,r,a,l,d)=>{const m=t*3;s[m]=r,s[m+1]=a,s[m+2]=l,e[m]=xf(d.r),e[m+1]=xf(d.g),e[m+2]=xf(d.b)},ru=(s,e)=>{const t=new Float32Array(s*3),r=new Float32Array(s*3);for(let a=0;a<s;a+=1)e(a,t,r);return{positions:t,colors:r}},lT=(s,e)=>{const t={r:1,g:.28,b:.46};return ru(s,(r,a,l)=>{const d=Math.random()*Math.PI*2,m=Math.pow(Math.random(),.42),g=16*Math.pow(Math.sin(d),3),_=13*Math.cos(d)-5*Math.cos(2*d)-2*Math.cos(3*d)-Math.cos(4*d),M=g*m*5.2+vi(-1.4,1.4),u=(_-2.2)*m*5.2+vi(-1.4,1.4),f=Math.sin(d*2)*7*m+vi(-6,6),p=to(t,e,.34+Math.random()*.22);sa(a,l,r,M,u,f,p)})},uT=(s,e)=>{const t={r:1,g:.55,b:.68},r={r:1,g:.82,b:.36};return ru(s,(a,l,d)=>{const m=Math.random()*Math.PI*2,g=Math.abs(Math.sin(5*m)),_=(18+58*g)*Math.pow(Math.random(),.36),M=Math.cos(m)*_+vi(-1,1),u=Math.sin(m)*_+Math.sin(m*3)*5+vi(-1,1),f=Math.cos(m*5)*8*g+vi(-4,4),p=_<18?.85:.18,y=to(to(t,r,p),e,.28);sa(l,d,a,M,u,f,y)})},cT=(s,e)=>{const t={r:.86,g:.72,b:.52},r={r:.96,g:.86,b:.64};return ru(s,(a,l,d)=>{if(Math.random()<.48){const S=Math.random()*Math.PI*2,v=vi(54,112),A=Math.cos(S)*v,P=Math.sin(S)*v*.28,L=Math.sin(S)*20+vi(-2.5,2.5);sa(l,d,a,A,P,L,to(r,e,.3));return}const g=Math.random()*Math.PI*2,_=Math.acos(vi(-1,1)),M=42*Math.pow(Math.random(),.28),u=Math.sin(_)*Math.cos(g)*M,f=Math.cos(_)*M,p=Math.sin(_)*Math.sin(g)*M,y=.12*Math.sin((f+42)*.22),E=to(t,e,.18+y);sa(l,d,a,u,f,p,E)})},fT=(s,e)=>{const t={r:1,g:.95,b:.82};return ru(s,(r,a,l)=>{const d=r%5,g=[{x:-70,y:28,z:0},{x:66,y:36,z:-4},{x:-14,y:-28,z:12},{x:24,y:78,z:-12},{x:86,y:-42,z:8}][d],_=Math.random()*Math.PI*2,M=Math.acos(vi(-1,1)),u=vi(12,58)*Math.pow(Math.random(),.18),f=Math.random()<.22?1.35:1,p=g.x+Math.sin(M)*Math.cos(_)*u*f,y=g.y+Math.cos(M)*u*.78,E=g.z+Math.sin(M)*Math.sin(_)*u,S=to(t,e,.2+Math.random()*.55);sa(a,l,r,p,y,E,S)})},Qr=window.matchMedia("(max-width: 700px)").matches?24e3:64e3,Ci={r:1,g:.76,b:.42},dT=Lv.degToRad(82),Sf=.035,hT=.085,pT=.035,Ol=(s,e)=>s+Math.random()*(e-s),mT=s=>{const e=Math.min(1,Math.max(0,s));return e<.5?4*e*e*e:1-Math.pow(-2*e+2,3)/2},_T=s=>{const e=new Float32Array(s*3),t=new Float32Array(s),r=new Float32Array(s);for(let a=0;a<s;a+=1){const l=Math.random()*Math.PI*2,d=Math.acos(Math.random()*2-1),m=72+Math.random()*130,g=a*3;e[g]=m*Math.sin(d)*Math.cos(l),e[g+1]=m*Math.sin(d)*Math.sin(l),e[g+2]=m*Math.cos(d),t[a]=Math.random()*1.9+1.1,r[a]=Math.random()*100}return{randoms:e,sizes:t,seeds:r}},$o=(s,e,t,r)=>{s.setAttribute(e,new Zt(t.slice(),r))},gT=({gesture:s,handRotation:e,model:t,themeColor:r})=>{const a=it.useRef(null),l=it.useRef(null),d=it.useRef(null),m=it.useRef(null),g=it.useRef(null),_=it.useRef(null),M=it.useRef(),u=it.useRef(null),f=it.useRef("heart"),p=it.useRef(null),y=it.useRef({start:0,duration:1200,active:!1}),E=it.useRef({start:-1e4,duration:1400,strength:0}),S=it.useRef({current:1,target:1}),v=it.useRef({current:0,target:0}),[A,P]=it.useState(!0),[L,z]=it.useState(!1);return it.useEffect(()=>{const D=a.current;if(!D)return;let F=null,R=null,I=null,W=!1;const O=q=>{const G=window.innerWidth,J=window.innerHeight,ie=G<=700;q.aspect=G/J,q.position.z=ie?Math.max(350,270/q.aspect):210,q.setViewOffset(G,J,ie?0:-135,ie?J*.16:0,G,J),q.updateProjectionMatrix()},j=async()=>{try{const q=new Xv;q.fog=new wd(197899,.006),l.current=q;const G=new ni(62,window.innerWidth/window.innerHeight,.1,1e3);O(G),d.current=G,I=new oT({antialias:!0,alpha:!0}),I.setSize(window.innerWidth,window.innerHeight),I.setPixelRatio(Math.min(window.devicePixelRatio,window.innerWidth<=700?1.5:2)),I.setClearColor(131850,1),D.appendChild(I.domElement),m.current=I;const J=lT(Qr,Ci),ie=uT(Qr,Ci),U=cT(Qr,Ci),K=fT(Qr,Ci),{randoms:Le,sizes:De,seeds:we}=_T(Qr),se=new Float32Array(Le),_e=new Float32Array(Qr*3);for(let $e=0;$e<Qr;$e++)_e[$e*3]=Ci.r,_e[$e*3+1]=Ci.g,_e[$e*3+2]=Ci.b;const de={positions:se,colors:_e};if(W)return;u.current={heart:J,flower:ie,saturn:U,fireworks:K},p.current=de,F=new ri,F.setAttribute("position",new Zt(de.positions.slice(),3)),F.setAttribute("posFrom",new Zt(de.positions.slice(),3)),F.setAttribute("posTo",new Zt(de.positions.slice(),3)),F.setAttribute("colorFrom",new Zt(de.colors.slice(),3)),F.setAttribute("colorTo",new Zt(de.colors.slice(),3)),F.setAttribute("randoms",new Zt(Le,3)),F.setAttribute("size",new Zt(De,1)),F.setAttribute("seed",new Zt(we,1)),S.current={current:1,target:1},R=new xi({uniforms:{time:{value:0},u_morph:{value:1},u_scatter:{value:1},u_themeColor:{value:new At(Ci.r,Ci.g,Ci.b)},u_accentStrength:{value:.48}},vertexShader:`
            uniform float time;
            uniform float u_morph;
            uniform float u_scatter;

            attribute vec3 posFrom;
            attribute vec3 posTo;
            attribute vec3 colorFrom;
            attribute vec3 colorTo;
            attribute vec3 randoms;
            attribute float size;
            attribute float seed;

            varying vec3 vColor;
            varying float vGlow;
            varying float vScatter;

            void main() {
              vec3 formed = mix(posFrom, posTo, u_morph);
              float drift = sin(time * 0.75 + seed) * 0.9 + cos(time * 0.42 + seed * 0.7) * 0.7;

              // Only apply breath and swirl to the scattered state (u_scatter > 0)
              // This makes the formed text perfectly stable and sharp
              vec3 breath = vec3(
                sin(time * 0.5 + seed) * 0.9,
                cos(time * 0.48 + seed * 1.3) * 0.8,
                drift
              ) * u_scatter;

              vec3 swirl = normalize(vec3(-formed.y, formed.x, randoms.z * 0.22) + 0.0001) * u_scatter * 16.0;
              vec3 scattered = formed + randoms * u_scatter + swirl + breath;

              vec4 mvPosition = modelViewMatrix * vec4(scattered, 1.0);
              float depth = max(24.0, length(mvPosition.xyz));

              // Sparkle effect also only applies to scattered particles, text should be solid
              float sparkle = mix(1.0, 0.72 + 0.28 * sin(time * 2.2 + seed * 6.2831), u_scatter);

              vColor = mix(colorFrom, colorTo, u_morph);
              vGlow = sparkle + u_scatter * 0.35;
              vScatter = u_scatter;

              // Base size for text is smaller to increase sharpness
              float baseSize = mix(size * 0.28, size, u_scatter);

              gl_PointSize = baseSize * sparkle * (360.0 / depth) * (1.0 + u_scatter * 0.55);
              gl_PointSize = clamp(gl_PointSize, 1.0, 6.0);
              gl_Position = projectionMatrix * mvPosition;
            }
          `,fragmentShader:`
            uniform vec3 u_themeColor;
            uniform float u_accentStrength;

            varying vec3 vColor;
            varying float vGlow;
            varying float vScatter;

            void main() {
              vec2 uv = gl_PointCoord.xy - vec2(0.5);
              float dist = length(uv);
              if (dist > 0.5) discard;

              float core = smoothstep(0.46, 0.08, dist);
              float halo = smoothstep(0.5, 0.0, dist) * mix(0.08, 0.45, vScatter);
              vec3 warmed = mix(vColor, u_themeColor, u_accentStrength * (0.22 + halo));
              vec3 finalColor = warmed * mix(1.06, 1.1 + vGlow * 0.58, vScatter);
              float alpha = (core * mix(0.98, 0.86, vScatter) + halo * mix(0.12, 0.34, vScatter)) * 0.94;

              gl_FragColor = vec4(finalColor, alpha);
            }
          `,transparent:!0,depthWrite:!1,blending:Xl});const Ie=new Cm(F,R);Ie.frustumCulled=!1,q.add(Ie),g.current=Ie;const je=re();q.add(je),_.current=je,P(!1)}catch{z(!0),P(!1)}},re=()=>{const G=new Float32Array(5100),J=new Float32Array(1700*3),ie=new Float32Array(1700),U=new At("#f8dfaa");for(let De=0;De<1700;De+=1){const we=De*3;G[we]=(Math.random()-.5)*420,G[we+1]=(Math.random()-.5)*250,G[we+2]=-90-Math.random()*190,J[we]=U.r*Ol(.58,1),J[we+1]=U.g*Ol(.58,1),J[we+2]=U.b*Ol(.58,1),ie[De]=Ol(.4,1.3)}const K=new ri;K.setAttribute("position",new Zt(G,3)),K.setAttribute("color",new Zt(J,3)),K.setAttribute("size",new Zt(ie,1));const Le=new D_({size:.8,vertexColors:!0,transparent:!0,opacity:.54,depthWrite:!1,blending:Xl});return new Cm(K,Le)};j();const ae=()=>{!d.current||!m.current||(O(d.current),m.current.setSize(window.innerWidth,window.innerHeight))};window.addEventListener("resize",ae);const X=performance.now(),Z=()=>{if(document.hidden){M.current=requestAnimationFrame(Z);return}const q=performance.now(),G=(q-X)/1e3;if(g.current){const J=g.current.material,ie=y.current;if(ie.active){const se=(q-ie.start)/ie.duration;J.uniforms.u_morph.value=mT(se),se>=1&&(J.uniforms.u_morph.value=1,ie.active=!1)}const U=E.current,K=(q-U.start)/U.duration,Le=K>=0&&K<=1?Math.sin(Math.PI*K):0,De=S.current;De.current+=(De.target-De.current)*pT,J.uniforms.time.value=G,J.uniforms.u_scatter.value=Math.min(1.48,De.current+Math.pow(Le,1.45)*U.strength);const we=l.current;if(we){const se=v.current;se.current+=(se.target-se.current)*hT,we.rotation.y=Math.sin(G*.16)*.09+se.current,we.rotation.x=Math.cos(G*.12)*.045}}_.current&&(_.current.rotation.z=G*.006,_.current.rotation.y=Math.sin(G*.08)*.05),m.current&&l.current&&d.current&&m.current.render(l.current,d.current),M.current=requestAnimationFrame(Z)};return Z(),()=>{var q,G,J;W=!0,window.removeEventListener("resize",ae),M.current&&cancelAnimationFrame(M.current),I&&D.contains(I.domElement)&&D.removeChild(I.domElement),F==null||F.dispose(),R==null||R.dispose(),(q=_.current)==null||q.geometry.dispose(),(J=(G=_.current)==null?void 0:G.material)==null||J.dispose(),I==null||I.dispose()}},[]),it.useEffect(()=>{if(e===null||Math.abs(e)<Sf){v.current.target=0;return}const D=(Math.abs(e)-Sf)/(1-Sf);v.current.target=Math.sign(e)*D*dT},[e,t]),it.useEffect(()=>{var R;const D=aT(r),F=(R=g.current)==null?void 0:R.material;F&&F.uniforms.u_themeColor.value.setRGB(D.r,D.g,D.b)},[r]),it.useEffect(()=>{const D=u.current,F=g.current;if(!D||!F)return;const R=t;if(s==="none")return;const I={positions:F.geometry.attributes.randoms.array,colors:F.geometry.attributes.colorFrom.array},W=s==="open",O=W?I:D[R],j=p.current??I;if(O===p.current)return;S.current.target=W?1:0;const re=F.geometry;$o(re,"posFrom",j.positions,3),$o(re,"posTo",O.positions,3),$o(re,"colorFrom",j.colors,3),$o(re,"colorTo",O.colors,3),$o(re,"position",O.positions,3),y.current={start:performance.now(),duration:W?3e3:2200,active:!0},E.current={start:performance.now(),duration:1500,strength:s==="fist"?.45:0},f.current=R,p.current=O},[s,t]),ut.jsxs(ut.Fragment,{children:[L&&ut.jsxs("div",{className:"render-error",role:"alert",children:["当前浏览器无法启动粒子画面，请使用支持 WebGL 的浏览器。",ut.jsx("br",{}),"Unable to start WebGL. Please try a compatible browser.",ut.jsx("br",{}),ut.jsx("a",{href:"../",children:"← Playground"})]}),A&&ut.jsx("div",{className:"pointer-events-none absolute inset-x-0 top-1/2 z-20 flex -translate-y-1/2 items-center justify-center",children:ut.jsx("div",{className:"rounded-full border border-amber-200/20 bg-black/30 px-5 py-3 text-sm font-medium tracking-[0.24em] text-amber-100/80 shadow-[0_0_48px_rgba(245,198,104,0.18)] backdrop-blur-xl",children:"WEAVING PARTICLES"})}),ut.jsx("div",{ref:a,className:"pointer-events-none absolute inset-0 z-0 bg-[#02030a]"})]})},vT="modulepreload",xT=function(s,e){return new URL(s,e).href},r_={},ST=function(e,t,r){let a=Promise.resolve();if(t&&t.length>0){let d=function(M){return Promise.all(M.map(u=>Promise.resolve(u).then(f=>({status:"fulfilled",value:f}),f=>({status:"rejected",reason:f}))))};const m=document.getElementsByTagName("link"),g=document.querySelector("meta[property=csp-nonce]"),_=(g==null?void 0:g.nonce)||(g==null?void 0:g.getAttribute("nonce"));a=d(t.map(M=>{if(M=xT(M,r),M in r_)return;r_[M]=!0;const u=M.endsWith(".css"),f=u?'[rel="stylesheet"]':"";if(!!r)for(let E=m.length-1;E>=0;E--){const S=m[E];if(S.href===M&&(!u||S.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${M}"]${f}`))return;const y=document.createElement("link");if(y.rel=u?"stylesheet":vT,u||(y.as="script"),y.crossOrigin="",y.href=M,_&&y.setAttribute("nonce",_),document.head.appendChild(y),u)return new Promise((E,S)=>{y.addEventListener("load",E),y.addEventListener("error",()=>S(new Error(`Unable to preload CSS for ${M}`)))})}))}function l(d){const m=new Event("vite:preloadError",{cancelable:!0});if(m.payload=d,window.dispatchEvent(m),!m.defaultPrevented)throw d}return a.then(d=>{for(const m of d||[])m.status==="rejected"&&l(m.reason);return e().catch(l)})},yT=.34,MT=.018,ET={fist:5,open:7,none:10},rs=(s,e,t)=>Math.min(t,Math.max(e,s)),TT=s=>{const e=rs(s,-1,1);return Math.sign(e)*Math.pow(Math.abs(e),.58)},Jo=(s,e)=>{const t=(s.z??0)-(e.z??0);return Math.hypot(s.x-e.x,s.y-e.y,t)},wT=s=>{const e=[0,5,9,13,17];return e.reduce((r,a)=>r+s[a].x,0)/e.length},s_=(s,e)=>({x:s.x-e.x,y:s.y-e.y,z:(s.z??0)-(e.z??0)}),AT=(s,e)=>({x:s.y*(e.z??0)-(s.z??0)*e.y,y:(s.z??0)*e.x-s.x*(e.z??0),z:s.x*e.y-s.y*e.x}),RT=s=>{const e=s[0],t=s_(s[5],e),r=s_(s[17],e),a=AT(t,r),l=Math.hypot(a.x,a.z??0);return l<1e-6?0:rs(a.x/l,-1,1)},CT=(s,e)=>{const t=e??s,r=t[5],a=t[17],l=Math.max(.001,Jo(r,a)),d=rs(((a.z??0)-(r.z??0))/l,-1,1),m=RT(t),g=rs((a.y-r.y)/l,-1,1);return rs(d*1.15+m*.9+g*.22,-1,1)},bT=(s,e)=>{const t=rs((.5-wT(s))*2,-1,1),r=CT(s,e);return TT(rs(t*.2+r*1.45,-1,1))},Bl=(s,e,t,r)=>{const a=s[0],l=Jo(s[e],a),d=Jo(s[t],a),m=Jo(s[e],s[r]),g=Jo(s[t],s[r]);return l>d*1.08&&m>g*1.15},PT=s=>{const e=[Bl(s,8,6,5),Bl(s,12,10,9),Bl(s,16,14,13),Bl(s,20,18,17)].filter(Boolean).length;return e>=3?"open":e<=1?"fist":"none"},LT=(s=!1)=>{const[e,t]=it.useState("none"),[r,a]=it.useState(null),[l,d]=it.useState(!1),[m,g]=it.useState(!1),_=it.useRef(null),M=it.useRef("none"),u=it.useRef("none"),f=it.useRef(0),p=it.useRef(null),y=it.useRef(null),E=it.useCallback(v=>{if(v===M.current){u.current=v,f.current=0;return}if(v!==u.current){u.current=v,f.current=1;return}f.current+=1,!(f.current<ET[v])&&(M.current=v,f.current=0,t(v))},[]),S=it.useCallback(v=>{if(v===null){p.current=null,y.current=null,a(null);return}const A=p.current,P=A===null?v:A+(v-A)*yT;p.current=P;const L=y.current;(L===null||Math.abs(L-P)>MT)&&(y.current=P,a(P))},[]);return it.useEffect(()=>{if(d(!1),g(!1),t("none"),a(null),M.current="none",u.current="none",f.current=0,p.current=null,y.current=null,!s||!_.current)return;const v=_.current;let A=!1,P,L,z=0;const D=()=>P==null?void 0:P.getTracks().forEach(W=>W.stop()),F=()=>{A=!0,cancelAnimationFrame(z),D(),v.srcObject=null,L==null||L.close().catch(()=>{})},R=()=>{A||(g(!0),d(!1),F())},I=async()=>{try{if(P=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:"user"},audio:!1}),A){D();return}v.srcObject=P,await v.play();const W=await ST(()=>import("./hands.js").then(re=>re.h),[],import.meta.url);if(A)return;L=new W.Hands({locateFile:re=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${re}`}),L.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:.64,minTrackingConfidence:.64}),L.onResults(re=>{var X,Z;if(A)return;const ae=(X=re.multiHandLandmarks)==null?void 0:X[0];ae?(S(bT(ae,(Z=re.multiHandWorldLandmarks)==null?void 0:Z[0])),E(PT(ae))):(S(null),E("none"))});let O;try{await Promise.race([L.initialize(),new Promise((re,ae)=>{O=setTimeout(()=>ae(new Error("Model load timed out")),3e4)})])}finally{clearTimeout(O)}if(A){L.close().catch(()=>{});return}d(!0);const j=async()=>{if(!A)try{!document.hidden&&v.readyState>=2&&await L.send({image:v}),A||(z=requestAnimationFrame(j))}catch{R()}};j()}catch{R()}};return window.addEventListener("pagehide",F),I(),()=>{window.removeEventListener("pagehide",F),F()}},[s,E,S]),{gesture:e,handRotation:r,isReady:l,error:m,videoRef:_}};var yf={};/*!
 *  howler.js v2.2.4
 *  howlerjs.com
 *
 *  (c) 2013-2020, James Simpson of GoldFire Studios
 *  goldfirestudios.com
 *
 *  MIT License
 */var o_;function DT(){return o_||(o_=1,(function(s){(function(){var e=function(){this.init()};e.prototype={init:function(){var u=this||t;return u._counter=1e3,u._html5AudioPool=[],u.html5PoolSize=10,u._codecs={},u._howls=[],u._muted=!1,u._volume=1,u._canPlayEvent="canplaythrough",u._navigator=typeof window<"u"&&window.navigator?window.navigator:null,u.masterGain=null,u.noAudio=!1,u.usingWebAudio=!0,u.autoSuspend=!0,u.ctx=null,u.autoUnlock=!0,u._setup(),u},volume:function(u){var f=this||t;if(u=parseFloat(u),f.ctx||M(),typeof u<"u"&&u>=0&&u<=1){if(f._volume=u,f._muted)return f;f.usingWebAudio&&f.masterGain.gain.setValueAtTime(u,t.ctx.currentTime);for(var p=0;p<f._howls.length;p++)if(!f._howls[p]._webAudio)for(var y=f._howls[p]._getSoundIds(),E=0;E<y.length;E++){var S=f._howls[p]._soundById(y[E]);S&&S._node&&(S._node.volume=S._volume*u)}return f}return f._volume},mute:function(u){var f=this||t;f.ctx||M(),f._muted=u,f.usingWebAudio&&f.masterGain.gain.setValueAtTime(u?0:f._volume,t.ctx.currentTime);for(var p=0;p<f._howls.length;p++)if(!f._howls[p]._webAudio)for(var y=f._howls[p]._getSoundIds(),E=0;E<y.length;E++){var S=f._howls[p]._soundById(y[E]);S&&S._node&&(S._node.muted=u?!0:S._muted)}return f},stop:function(){for(var u=this||t,f=0;f<u._howls.length;f++)u._howls[f].stop();return u},unload:function(){for(var u=this||t,f=u._howls.length-1;f>=0;f--)u._howls[f].unload();return u.usingWebAudio&&u.ctx&&typeof u.ctx.close<"u"&&(u.ctx.close(),u.ctx=null,M()),u},codecs:function(u){return(this||t)._codecs[u.replace(/^x-/,"")]},_setup:function(){var u=this||t;if(u.state=u.ctx&&u.ctx.state||"suspended",u._autoSuspend(),!u.usingWebAudio)if(typeof Audio<"u")try{var f=new Audio;typeof f.oncanplaythrough>"u"&&(u._canPlayEvent="canplay")}catch{u.noAudio=!0}else u.noAudio=!0;try{var f=new Audio;f.muted&&(u.noAudio=!0)}catch{}return u.noAudio||u._setupCodecs(),u},_setupCodecs:function(){var u=this||t,f=null;try{f=typeof Audio<"u"?new Audio:null}catch{return u}if(!f||typeof f.canPlayType!="function")return u;var p=f.canPlayType("audio/mpeg;").replace(/^no$/,""),y=u._navigator?u._navigator.userAgent:"",E=y.match(/OPR\/(\d+)/g),S=E&&parseInt(E[0].split("/")[1],10)<33,v=y.indexOf("Safari")!==-1&&y.indexOf("Chrome")===-1,A=y.match(/Version\/(.*?) /),P=v&&A&&parseInt(A[1],10)<15;return u._codecs={mp3:!!(!S&&(p||f.canPlayType("audio/mp3;").replace(/^no$/,""))),mpeg:!!p,opus:!!f.canPlayType('audio/ogg; codecs="opus"').replace(/^no$/,""),ogg:!!f.canPlayType('audio/ogg; codecs="vorbis"').replace(/^no$/,""),oga:!!f.canPlayType('audio/ogg; codecs="vorbis"').replace(/^no$/,""),wav:!!(f.canPlayType('audio/wav; codecs="1"')||f.canPlayType("audio/wav")).replace(/^no$/,""),aac:!!f.canPlayType("audio/aac;").replace(/^no$/,""),caf:!!f.canPlayType("audio/x-caf;").replace(/^no$/,""),m4a:!!(f.canPlayType("audio/x-m4a;")||f.canPlayType("audio/m4a;")||f.canPlayType("audio/aac;")).replace(/^no$/,""),m4b:!!(f.canPlayType("audio/x-m4b;")||f.canPlayType("audio/m4b;")||f.canPlayType("audio/aac;")).replace(/^no$/,""),mp4:!!(f.canPlayType("audio/x-mp4;")||f.canPlayType("audio/mp4;")||f.canPlayType("audio/aac;")).replace(/^no$/,""),weba:!!(!P&&f.canPlayType('audio/webm; codecs="vorbis"').replace(/^no$/,"")),webm:!!(!P&&f.canPlayType('audio/webm; codecs="vorbis"').replace(/^no$/,"")),dolby:!!f.canPlayType('audio/mp4; codecs="ec-3"').replace(/^no$/,""),flac:!!(f.canPlayType("audio/x-flac;")||f.canPlayType("audio/flac;")).replace(/^no$/,"")},u},_unlockAudio:function(){var u=this||t;if(!(u._audioUnlocked||!u.ctx)){u._audioUnlocked=!1,u.autoUnlock=!1,!u._mobileUnloaded&&u.ctx.sampleRate!==44100&&(u._mobileUnloaded=!0,u.unload()),u._scratchBuffer=u.ctx.createBuffer(1,1,22050);var f=function(p){for(;u._html5AudioPool.length<u.html5PoolSize;)try{var y=new Audio;y._unlocked=!0,u._releaseHtml5Audio(y)}catch{u.noAudio=!0;break}for(var E=0;E<u._howls.length;E++)if(!u._howls[E]._webAudio)for(var S=u._howls[E]._getSoundIds(),v=0;v<S.length;v++){var A=u._howls[E]._soundById(S[v]);A&&A._node&&!A._node._unlocked&&(A._node._unlocked=!0,A._node.load())}u._autoResume();var P=u.ctx.createBufferSource();P.buffer=u._scratchBuffer,P.connect(u.ctx.destination),typeof P.start>"u"?P.noteOn(0):P.start(0),typeof u.ctx.resume=="function"&&u.ctx.resume(),P.onended=function(){P.disconnect(0),u._audioUnlocked=!0,document.removeEventListener("touchstart",f,!0),document.removeEventListener("touchend",f,!0),document.removeEventListener("click",f,!0),document.removeEventListener("keydown",f,!0);for(var L=0;L<u._howls.length;L++)u._howls[L]._emit("unlock")}};return document.addEventListener("touchstart",f,!0),document.addEventListener("touchend",f,!0),document.addEventListener("click",f,!0),document.addEventListener("keydown",f,!0),u}},_obtainHtml5Audio:function(){var u=this||t;if(u._html5AudioPool.length)return u._html5AudioPool.pop();var f=new Audio().play();return f&&typeof Promise<"u"&&(f instanceof Promise||typeof f.then=="function")&&f.catch(function(){console.warn("HTML5 Audio pool exhausted, returning potentially locked audio object.")}),new Audio},_releaseHtml5Audio:function(u){var f=this||t;return u._unlocked&&f._html5AudioPool.push(u),f},_autoSuspend:function(){var u=this;if(!(!u.autoSuspend||!u.ctx||typeof u.ctx.suspend>"u"||!t.usingWebAudio)){for(var f=0;f<u._howls.length;f++)if(u._howls[f]._webAudio){for(var p=0;p<u._howls[f]._sounds.length;p++)if(!u._howls[f]._sounds[p]._paused)return u}return u._suspendTimer&&clearTimeout(u._suspendTimer),u._suspendTimer=setTimeout(function(){if(u.autoSuspend){u._suspendTimer=null,u.state="suspending";var y=function(){u.state="suspended",u._resumeAfterSuspend&&(delete u._resumeAfterSuspend,u._autoResume())};u.ctx.suspend().then(y,y)}},3e4),u}},_autoResume:function(){var u=this;if(!(!u.ctx||typeof u.ctx.resume>"u"||!t.usingWebAudio))return u.state==="running"&&u.ctx.state!=="interrupted"&&u._suspendTimer?(clearTimeout(u._suspendTimer),u._suspendTimer=null):u.state==="suspended"||u.state==="running"&&u.ctx.state==="interrupted"?(u.ctx.resume().then(function(){u.state="running";for(var f=0;f<u._howls.length;f++)u._howls[f]._emit("resume")}),u._suspendTimer&&(clearTimeout(u._suspendTimer),u._suspendTimer=null)):u.state==="suspending"&&(u._resumeAfterSuspend=!0),u}};var t=new e,r=function(u){var f=this;if(!u.src||u.src.length===0){console.error("An array of source files must be passed with any new Howl.");return}f.init(u)};r.prototype={init:function(u){var f=this;return t.ctx||M(),f._autoplay=u.autoplay||!1,f._format=typeof u.format!="string"?u.format:[u.format],f._html5=u.html5||!1,f._muted=u.mute||!1,f._loop=u.loop||!1,f._pool=u.pool||5,f._preload=typeof u.preload=="boolean"||u.preload==="metadata"?u.preload:!0,f._rate=u.rate||1,f._sprite=u.sprite||{},f._src=typeof u.src!="string"?u.src:[u.src],f._volume=u.volume!==void 0?u.volume:1,f._xhr={method:u.xhr&&u.xhr.method?u.xhr.method:"GET",headers:u.xhr&&u.xhr.headers?u.xhr.headers:null,withCredentials:u.xhr&&u.xhr.withCredentials?u.xhr.withCredentials:!1},f._duration=0,f._state="unloaded",f._sounds=[],f._endTimers={},f._queue=[],f._playLock=!1,f._onend=u.onend?[{fn:u.onend}]:[],f._onfade=u.onfade?[{fn:u.onfade}]:[],f._onload=u.onload?[{fn:u.onload}]:[],f._onloaderror=u.onloaderror?[{fn:u.onloaderror}]:[],f._onplayerror=u.onplayerror?[{fn:u.onplayerror}]:[],f._onpause=u.onpause?[{fn:u.onpause}]:[],f._onplay=u.onplay?[{fn:u.onplay}]:[],f._onstop=u.onstop?[{fn:u.onstop}]:[],f._onmute=u.onmute?[{fn:u.onmute}]:[],f._onvolume=u.onvolume?[{fn:u.onvolume}]:[],f._onrate=u.onrate?[{fn:u.onrate}]:[],f._onseek=u.onseek?[{fn:u.onseek}]:[],f._onunlock=u.onunlock?[{fn:u.onunlock}]:[],f._onresume=[],f._webAudio=t.usingWebAudio&&!f._html5,typeof t.ctx<"u"&&t.ctx&&t.autoUnlock&&t._unlockAudio(),t._howls.push(f),f._autoplay&&f._queue.push({event:"play",action:function(){f.play()}}),f._preload&&f._preload!=="none"&&f.load(),f},load:function(){var u=this,f=null;if(t.noAudio){u._emit("loaderror",null,"No audio support.");return}typeof u._src=="string"&&(u._src=[u._src]);for(var p=0;p<u._src.length;p++){var y,E;if(u._format&&u._format[p])y=u._format[p];else{if(E=u._src[p],typeof E!="string"){u._emit("loaderror",null,"Non-string found in selected audio sources - ignoring.");continue}y=/^data:audio\/([^;,]+);/i.exec(E),y||(y=/\.([^.]+)$/.exec(E.split("?",1)[0])),y&&(y=y[1].toLowerCase())}if(y||console.warn('No file extension was found. Consider using the "format" property or specify an extension.'),y&&t.codecs(y)){f=u._src[p];break}}if(!f){u._emit("loaderror",null,"No codec support for selected audio sources.");return}return u._src=f,u._state="loading",window.location.protocol==="https:"&&f.slice(0,5)==="http:"&&(u._html5=!0,u._webAudio=!1),new a(u),u._webAudio&&d(u),u},play:function(u,f){var p=this,y=null;if(typeof u=="number")y=u,u=null;else{if(typeof u=="string"&&p._state==="loaded"&&!p._sprite[u])return null;if(typeof u>"u"&&(u="__default",!p._playLock)){for(var E=0,S=0;S<p._sounds.length;S++)p._sounds[S]._paused&&!p._sounds[S]._ended&&(E++,y=p._sounds[S]._id);E===1?u=null:y=null}}var v=y?p._soundById(y):p._inactiveSound();if(!v)return null;if(y&&!u&&(u=v._sprite||"__default"),p._state!=="loaded"){v._sprite=u,v._ended=!1;var A=v._id;return p._queue.push({event:"play",action:function(){p.play(A)}}),A}if(y&&!v._paused)return f||p._loadQueue("play"),v._id;p._webAudio&&t._autoResume();var P=Math.max(0,v._seek>0?v._seek:p._sprite[u][0]/1e3),L=Math.max(0,(p._sprite[u][0]+p._sprite[u][1])/1e3-P),z=L*1e3/Math.abs(v._rate),D=p._sprite[u][0]/1e3,F=(p._sprite[u][0]+p._sprite[u][1])/1e3;v._sprite=u,v._ended=!1;var R=function(){v._paused=!1,v._seek=P,v._start=D,v._stop=F,v._loop=!!(v._loop||p._sprite[u][2])};if(P>=F){p._ended(v);return}var I=v._node;if(p._webAudio){var W=function(){p._playLock=!1,R(),p._refreshBuffer(v);var ae=v._muted||p._muted?0:v._volume;I.gain.setValueAtTime(ae,t.ctx.currentTime),v._playStart=t.ctx.currentTime,typeof I.bufferSource.start>"u"?v._loop?I.bufferSource.noteGrainOn(0,P,86400):I.bufferSource.noteGrainOn(0,P,L):v._loop?I.bufferSource.start(0,P,86400):I.bufferSource.start(0,P,L),z!==1/0&&(p._endTimers[v._id]=setTimeout(p._ended.bind(p,v),z)),f||setTimeout(function(){p._emit("play",v._id),p._loadQueue()},0)};t.state==="running"&&t.ctx.state!=="interrupted"?W():(p._playLock=!0,p.once("resume",W),p._clearTimer(v._id))}else{var O=function(){I.currentTime=P,I.muted=v._muted||p._muted||t._muted||I.muted,I.volume=v._volume*t.volume(),I.playbackRate=v._rate;try{var ae=I.play();if(ae&&typeof Promise<"u"&&(ae instanceof Promise||typeof ae.then=="function")?(p._playLock=!0,R(),ae.then(function(){p._playLock=!1,I._unlocked=!0,f?p._loadQueue():p._emit("play",v._id)}).catch(function(){p._playLock=!1,p._emit("playerror",v._id,"Playback was unable to start. This is most commonly an issue on mobile devices and Chrome where playback was not within a user interaction."),v._ended=!0,v._paused=!0})):f||(p._playLock=!1,R(),p._emit("play",v._id)),I.playbackRate=v._rate,I.paused){p._emit("playerror",v._id,"Playback was unable to start. This is most commonly an issue on mobile devices and Chrome where playback was not within a user interaction.");return}u!=="__default"||v._loop?p._endTimers[v._id]=setTimeout(p._ended.bind(p,v),z):(p._endTimers[v._id]=function(){p._ended(v),I.removeEventListener("ended",p._endTimers[v._id],!1)},I.addEventListener("ended",p._endTimers[v._id],!1))}catch(X){p._emit("playerror",v._id,X)}};I.src==="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"&&(I.src=p._src,I.load());var j=window&&window.ejecta||!I.readyState&&t._navigator.isCocoonJS;if(I.readyState>=3||j)O();else{p._playLock=!0,p._state="loading";var re=function(){p._state="loaded",O(),I.removeEventListener(t._canPlayEvent,re,!1)};I.addEventListener(t._canPlayEvent,re,!1),p._clearTimer(v._id)}}return v._id},pause:function(u){var f=this;if(f._state!=="loaded"||f._playLock)return f._queue.push({event:"pause",action:function(){f.pause(u)}}),f;for(var p=f._getSoundIds(u),y=0;y<p.length;y++){f._clearTimer(p[y]);var E=f._soundById(p[y]);if(E&&!E._paused&&(E._seek=f.seek(p[y]),E._rateSeek=0,E._paused=!0,f._stopFade(p[y]),E._node))if(f._webAudio){if(!E._node.bufferSource)continue;typeof E._node.bufferSource.stop>"u"?E._node.bufferSource.noteOff(0):E._node.bufferSource.stop(0),f._cleanBuffer(E._node)}else(!isNaN(E._node.duration)||E._node.duration===1/0)&&E._node.pause();arguments[1]||f._emit("pause",E?E._id:null)}return f},stop:function(u,f){var p=this;if(p._state!=="loaded"||p._playLock)return p._queue.push({event:"stop",action:function(){p.stop(u)}}),p;for(var y=p._getSoundIds(u),E=0;E<y.length;E++){p._clearTimer(y[E]);var S=p._soundById(y[E]);S&&(S._seek=S._start||0,S._rateSeek=0,S._paused=!0,S._ended=!0,p._stopFade(y[E]),S._node&&(p._webAudio?S._node.bufferSource&&(typeof S._node.bufferSource.stop>"u"?S._node.bufferSource.noteOff(0):S._node.bufferSource.stop(0),p._cleanBuffer(S._node)):(!isNaN(S._node.duration)||S._node.duration===1/0)&&(S._node.currentTime=S._start||0,S._node.pause(),S._node.duration===1/0&&p._clearSound(S._node))),f||p._emit("stop",S._id))}return p},mute:function(u,f){var p=this;if(p._state!=="loaded"||p._playLock)return p._queue.push({event:"mute",action:function(){p.mute(u,f)}}),p;if(typeof f>"u")if(typeof u=="boolean")p._muted=u;else return p._muted;for(var y=p._getSoundIds(f),E=0;E<y.length;E++){var S=p._soundById(y[E]);S&&(S._muted=u,S._interval&&p._stopFade(S._id),p._webAudio&&S._node?S._node.gain.setValueAtTime(u?0:S._volume,t.ctx.currentTime):S._node&&(S._node.muted=t._muted?!0:u),p._emit("mute",S._id))}return p},volume:function(){var u=this,f=arguments,p,y;if(f.length===0)return u._volume;if(f.length===1||f.length===2&&typeof f[1]>"u"){var E=u._getSoundIds(),S=E.indexOf(f[0]);S>=0?y=parseInt(f[0],10):p=parseFloat(f[0])}else f.length>=2&&(p=parseFloat(f[0]),y=parseInt(f[1],10));var v;if(typeof p<"u"&&p>=0&&p<=1){if(u._state!=="loaded"||u._playLock)return u._queue.push({event:"volume",action:function(){u.volume.apply(u,f)}}),u;typeof y>"u"&&(u._volume=p),y=u._getSoundIds(y);for(var A=0;A<y.length;A++)v=u._soundById(y[A]),v&&(v._volume=p,f[2]||u._stopFade(y[A]),u._webAudio&&v._node&&!v._muted?v._node.gain.setValueAtTime(p,t.ctx.currentTime):v._node&&!v._muted&&(v._node.volume=p*t.volume()),u._emit("volume",v._id))}else return v=y?u._soundById(y):u._sounds[0],v?v._volume:0;return u},fade:function(u,f,p,y){var E=this;if(E._state!=="loaded"||E._playLock)return E._queue.push({event:"fade",action:function(){E.fade(u,f,p,y)}}),E;u=Math.min(Math.max(0,parseFloat(u)),1),f=Math.min(Math.max(0,parseFloat(f)),1),p=parseFloat(p),E.volume(u,y);for(var S=E._getSoundIds(y),v=0;v<S.length;v++){var A=E._soundById(S[v]);if(A){if(y||E._stopFade(S[v]),E._webAudio&&!A._muted){var P=t.ctx.currentTime,L=P+p/1e3;A._volume=u,A._node.gain.setValueAtTime(u,P),A._node.gain.linearRampToValueAtTime(f,L)}E._startFadeInterval(A,u,f,p,S[v],typeof y>"u")}}return E},_startFadeInterval:function(u,f,p,y,E,S){var v=this,A=f,P=p-f,L=Math.abs(P/.01),z=Math.max(4,L>0?y/L:y),D=Date.now();u._fadeTo=p,u._interval=setInterval(function(){var F=(Date.now()-D)/y;D=Date.now(),A+=P*F,A=Math.round(A*100)/100,P<0?A=Math.max(p,A):A=Math.min(p,A),v._webAudio?u._volume=A:v.volume(A,u._id,!0),S&&(v._volume=A),(p<f&&A<=p||p>f&&A>=p)&&(clearInterval(u._interval),u._interval=null,u._fadeTo=null,v.volume(p,u._id),v._emit("fade",u._id))},z)},_stopFade:function(u){var f=this,p=f._soundById(u);return p&&p._interval&&(f._webAudio&&p._node.gain.cancelScheduledValues(t.ctx.currentTime),clearInterval(p._interval),p._interval=null,f.volume(p._fadeTo,u),p._fadeTo=null,f._emit("fade",u)),f},loop:function(){var u=this,f=arguments,p,y,E;if(f.length===0)return u._loop;if(f.length===1)if(typeof f[0]=="boolean")p=f[0],u._loop=p;else return E=u._soundById(parseInt(f[0],10)),E?E._loop:!1;else f.length===2&&(p=f[0],y=parseInt(f[1],10));for(var S=u._getSoundIds(y),v=0;v<S.length;v++)E=u._soundById(S[v]),E&&(E._loop=p,u._webAudio&&E._node&&E._node.bufferSource&&(E._node.bufferSource.loop=p,p&&(E._node.bufferSource.loopStart=E._start||0,E._node.bufferSource.loopEnd=E._stop,u.playing(S[v])&&(u.pause(S[v],!0),u.play(S[v],!0)))));return u},rate:function(){var u=this,f=arguments,p,y;if(f.length===0)y=u._sounds[0]._id;else if(f.length===1){var E=u._getSoundIds(),S=E.indexOf(f[0]);S>=0?y=parseInt(f[0],10):p=parseFloat(f[0])}else f.length===2&&(p=parseFloat(f[0]),y=parseInt(f[1],10));var v;if(typeof p=="number"){if(u._state!=="loaded"||u._playLock)return u._queue.push({event:"rate",action:function(){u.rate.apply(u,f)}}),u;typeof y>"u"&&(u._rate=p),y=u._getSoundIds(y);for(var A=0;A<y.length;A++)if(v=u._soundById(y[A]),v){u.playing(y[A])&&(v._rateSeek=u.seek(y[A]),v._playStart=u._webAudio?t.ctx.currentTime:v._playStart),v._rate=p,u._webAudio&&v._node&&v._node.bufferSource?v._node.bufferSource.playbackRate.setValueAtTime(p,t.ctx.currentTime):v._node&&(v._node.playbackRate=p);var P=u.seek(y[A]),L=(u._sprite[v._sprite][0]+u._sprite[v._sprite][1])/1e3-P,z=L*1e3/Math.abs(v._rate);(u._endTimers[y[A]]||!v._paused)&&(u._clearTimer(y[A]),u._endTimers[y[A]]=setTimeout(u._ended.bind(u,v),z)),u._emit("rate",v._id)}}else return v=u._soundById(y),v?v._rate:u._rate;return u},seek:function(){var u=this,f=arguments,p,y;if(f.length===0)u._sounds.length&&(y=u._sounds[0]._id);else if(f.length===1){var E=u._getSoundIds(),S=E.indexOf(f[0]);S>=0?y=parseInt(f[0],10):u._sounds.length&&(y=u._sounds[0]._id,p=parseFloat(f[0]))}else f.length===2&&(p=parseFloat(f[0]),y=parseInt(f[1],10));if(typeof y>"u")return 0;if(typeof p=="number"&&(u._state!=="loaded"||u._playLock))return u._queue.push({event:"seek",action:function(){u.seek.apply(u,f)}}),u;var v=u._soundById(y);if(v)if(typeof p=="number"&&p>=0){var A=u.playing(y);A&&u.pause(y,!0),v._seek=p,v._ended=!1,u._clearTimer(y),!u._webAudio&&v._node&&!isNaN(v._node.duration)&&(v._node.currentTime=p);var P=function(){A&&u.play(y,!0),u._emit("seek",y)};if(A&&!u._webAudio){var L=function(){u._playLock?setTimeout(L,0):P()};setTimeout(L,0)}else P()}else if(u._webAudio){var z=u.playing(y)?t.ctx.currentTime-v._playStart:0,D=v._rateSeek?v._rateSeek-v._seek:0;return v._seek+(D+z*Math.abs(v._rate))}else return v._node.currentTime;return u},playing:function(u){var f=this;if(typeof u=="number"){var p=f._soundById(u);return p?!p._paused:!1}for(var y=0;y<f._sounds.length;y++)if(!f._sounds[y]._paused)return!0;return!1},duration:function(u){var f=this,p=f._duration,y=f._soundById(u);return y&&(p=f._sprite[y._sprite][1]/1e3),p},state:function(){return this._state},unload:function(){for(var u=this,f=u._sounds,p=0;p<f.length;p++)f[p]._paused||u.stop(f[p]._id),u._webAudio||(u._clearSound(f[p]._node),f[p]._node.removeEventListener("error",f[p]._errorFn,!1),f[p]._node.removeEventListener(t._canPlayEvent,f[p]._loadFn,!1),f[p]._node.removeEventListener("ended",f[p]._endFn,!1),t._releaseHtml5Audio(f[p]._node)),delete f[p]._node,u._clearTimer(f[p]._id);var y=t._howls.indexOf(u);y>=0&&t._howls.splice(y,1);var E=!0;for(p=0;p<t._howls.length;p++)if(t._howls[p]._src===u._src||u._src.indexOf(t._howls[p]._src)>=0){E=!1;break}return l&&E&&delete l[u._src],t.noAudio=!1,u._state="unloaded",u._sounds=[],u=null,null},on:function(u,f,p,y){var E=this,S=E["_on"+u];return typeof f=="function"&&S.push(y?{id:p,fn:f,once:y}:{id:p,fn:f}),E},off:function(u,f,p){var y=this,E=y["_on"+u],S=0;if(typeof f=="number"&&(p=f,f=null),f||p)for(S=0;S<E.length;S++){var v=p===E[S].id;if(f===E[S].fn&&v||!f&&v){E.splice(S,1);break}}else if(u)y["_on"+u]=[];else{var A=Object.keys(y);for(S=0;S<A.length;S++)A[S].indexOf("_on")===0&&Array.isArray(y[A[S]])&&(y[A[S]]=[])}return y},once:function(u,f,p){var y=this;return y.on(u,f,p,1),y},_emit:function(u,f,p){for(var y=this,E=y["_on"+u],S=E.length-1;S>=0;S--)(!E[S].id||E[S].id===f||u==="load")&&(setTimeout((function(v){v.call(this,f,p)}).bind(y,E[S].fn),0),E[S].once&&y.off(u,E[S].fn,E[S].id));return y._loadQueue(u),y},_loadQueue:function(u){var f=this;if(f._queue.length>0){var p=f._queue[0];p.event===u&&(f._queue.shift(),f._loadQueue()),u||p.action()}return f},_ended:function(u){var f=this,p=u._sprite;if(!f._webAudio&&u._node&&!u._node.paused&&!u._node.ended&&u._node.currentTime<u._stop)return setTimeout(f._ended.bind(f,u),100),f;var y=!!(u._loop||f._sprite[p][2]);if(f._emit("end",u._id),!f._webAudio&&y&&f.stop(u._id,!0).play(u._id),f._webAudio&&y){f._emit("play",u._id),u._seek=u._start||0,u._rateSeek=0,u._playStart=t.ctx.currentTime;var E=(u._stop-u._start)*1e3/Math.abs(u._rate);f._endTimers[u._id]=setTimeout(f._ended.bind(f,u),E)}return f._webAudio&&!y&&(u._paused=!0,u._ended=!0,u._seek=u._start||0,u._rateSeek=0,f._clearTimer(u._id),f._cleanBuffer(u._node),t._autoSuspend()),!f._webAudio&&!y&&f.stop(u._id,!0),f},_clearTimer:function(u){var f=this;if(f._endTimers[u]){if(typeof f._endTimers[u]!="function")clearTimeout(f._endTimers[u]);else{var p=f._soundById(u);p&&p._node&&p._node.removeEventListener("ended",f._endTimers[u],!1)}delete f._endTimers[u]}return f},_soundById:function(u){for(var f=this,p=0;p<f._sounds.length;p++)if(u===f._sounds[p]._id)return f._sounds[p];return null},_inactiveSound:function(){var u=this;u._drain();for(var f=0;f<u._sounds.length;f++)if(u._sounds[f]._ended)return u._sounds[f].reset();return new a(u)},_drain:function(){var u=this,f=u._pool,p=0,y=0;if(!(u._sounds.length<f)){for(y=0;y<u._sounds.length;y++)u._sounds[y]._ended&&p++;for(y=u._sounds.length-1;y>=0;y--){if(p<=f)return;u._sounds[y]._ended&&(u._webAudio&&u._sounds[y]._node&&u._sounds[y]._node.disconnect(0),u._sounds.splice(y,1),p--)}}},_getSoundIds:function(u){var f=this;if(typeof u>"u"){for(var p=[],y=0;y<f._sounds.length;y++)p.push(f._sounds[y]._id);return p}else return[u]},_refreshBuffer:function(u){var f=this;return u._node.bufferSource=t.ctx.createBufferSource(),u._node.bufferSource.buffer=l[f._src],u._panner?u._node.bufferSource.connect(u._panner):u._node.bufferSource.connect(u._node),u._node.bufferSource.loop=u._loop,u._loop&&(u._node.bufferSource.loopStart=u._start||0,u._node.bufferSource.loopEnd=u._stop||0),u._node.bufferSource.playbackRate.setValueAtTime(u._rate,t.ctx.currentTime),f},_cleanBuffer:function(u){var f=this,p=t._navigator&&t._navigator.vendor.indexOf("Apple")>=0;if(!u.bufferSource)return f;if(t._scratchBuffer&&u.bufferSource&&(u.bufferSource.onended=null,u.bufferSource.disconnect(0),p))try{u.bufferSource.buffer=t._scratchBuffer}catch{}return u.bufferSource=null,f},_clearSound:function(u){var f=/MSIE |Trident\//.test(t._navigator&&t._navigator.userAgent);f||(u.src="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA")}};var a=function(u){this._parent=u,this.init()};a.prototype={init:function(){var u=this,f=u._parent;return u._muted=f._muted,u._loop=f._loop,u._volume=f._volume,u._rate=f._rate,u._seek=0,u._paused=!0,u._ended=!0,u._sprite="__default",u._id=++t._counter,f._sounds.push(u),u.create(),u},create:function(){var u=this,f=u._parent,p=t._muted||u._muted||u._parent._muted?0:u._volume;return f._webAudio?(u._node=typeof t.ctx.createGain>"u"?t.ctx.createGainNode():t.ctx.createGain(),u._node.gain.setValueAtTime(p,t.ctx.currentTime),u._node.paused=!0,u._node.connect(t.masterGain)):t.noAudio||(u._node=t._obtainHtml5Audio(),u._errorFn=u._errorListener.bind(u),u._node.addEventListener("error",u._errorFn,!1),u._loadFn=u._loadListener.bind(u),u._node.addEventListener(t._canPlayEvent,u._loadFn,!1),u._endFn=u._endListener.bind(u),u._node.addEventListener("ended",u._endFn,!1),u._node.src=f._src,u._node.preload=f._preload===!0?"auto":f._preload,u._node.volume=p*t.volume(),u._node.load()),u},reset:function(){var u=this,f=u._parent;return u._muted=f._muted,u._loop=f._loop,u._volume=f._volume,u._rate=f._rate,u._seek=0,u._rateSeek=0,u._paused=!0,u._ended=!0,u._sprite="__default",u._id=++t._counter,u},_errorListener:function(){var u=this;u._parent._emit("loaderror",u._id,u._node.error?u._node.error.code:0),u._node.removeEventListener("error",u._errorFn,!1)},_loadListener:function(){var u=this,f=u._parent;f._duration=Math.ceil(u._node.duration*10)/10,Object.keys(f._sprite).length===0&&(f._sprite={__default:[0,f._duration*1e3]}),f._state!=="loaded"&&(f._state="loaded",f._emit("load"),f._loadQueue()),u._node.removeEventListener(t._canPlayEvent,u._loadFn,!1)},_endListener:function(){var u=this,f=u._parent;f._duration===1/0&&(f._duration=Math.ceil(u._node.duration*10)/10,f._sprite.__default[1]===1/0&&(f._sprite.__default[1]=f._duration*1e3),f._ended(u)),u._node.removeEventListener("ended",u._endFn,!1)}};var l={},d=function(u){var f=u._src;if(l[f]){u._duration=l[f].duration,_(u);return}if(/^data:[^;]+;base64,/.test(f)){for(var p=atob(f.split(",")[1]),y=new Uint8Array(p.length),E=0;E<p.length;++E)y[E]=p.charCodeAt(E);g(y.buffer,u)}else{var S=new XMLHttpRequest;S.open(u._xhr.method,f,!0),S.withCredentials=u._xhr.withCredentials,S.responseType="arraybuffer",u._xhr.headers&&Object.keys(u._xhr.headers).forEach(function(v){S.setRequestHeader(v,u._xhr.headers[v])}),S.onload=function(){var v=(S.status+"")[0];if(v!=="0"&&v!=="2"&&v!=="3"){u._emit("loaderror",null,"Failed loading audio file with status: "+S.status+".");return}g(S.response,u)},S.onerror=function(){u._webAudio&&(u._html5=!0,u._webAudio=!1,u._sounds=[],delete l[f],u.load())},m(S)}},m=function(u){try{u.send()}catch{u.onerror()}},g=function(u,f){var p=function(){f._emit("loaderror",null,"Decoding audio data failed.")},y=function(E){E&&f._sounds.length>0?(l[f._src]=E,_(f,E)):p()};typeof Promise<"u"&&t.ctx.decodeAudioData.length===1?t.ctx.decodeAudioData(u).then(y).catch(p):t.ctx.decodeAudioData(u,y,p)},_=function(u,f){f&&!u._duration&&(u._duration=f.duration),Object.keys(u._sprite).length===0&&(u._sprite={__default:[0,u._duration*1e3]}),u._state!=="loaded"&&(u._state="loaded",u._emit("load"),u._loadQueue())},M=function(){if(t.usingWebAudio){try{typeof AudioContext<"u"?t.ctx=new AudioContext:typeof webkitAudioContext<"u"?t.ctx=new webkitAudioContext:t.usingWebAudio=!1}catch{t.usingWebAudio=!1}t.ctx||(t.usingWebAudio=!1);var u=/iP(hone|od|ad)/.test(t._navigator&&t._navigator.platform),f=t._navigator&&t._navigator.appVersion.match(/OS (\d+)_(\d+)_?(\d+)?/),p=f?parseInt(f[1],10):null;if(u&&p&&p<9){var y=/safari/.test(t._navigator&&t._navigator.userAgent.toLowerCase());t._navigator&&!y&&(t.usingWebAudio=!1)}t.usingWebAudio&&(t.masterGain=typeof t.ctx.createGain>"u"?t.ctx.createGainNode():t.ctx.createGain(),t.masterGain.gain.setValueAtTime(t._muted?0:t._volume,t.ctx.currentTime),t.masterGain.connect(t.ctx.destination)),t._setup()}};s.Howler=t,s.Howl=r,typeof Vo<"u"?(Vo.HowlerGlobal=e,Vo.Howler=t,Vo.Howl=r,Vo.Sound=a):typeof window<"u"&&(window.HowlerGlobal=e,window.Howler=t,window.Howl=r,window.Sound=a)})();/*!
 *  Spatial Plugin - Adds support for stereo and 3D audio where Web Audio is supported.
 *  
 *  howler.js v2.2.4
 *  howlerjs.com
 *
 *  (c) 2013-2020, James Simpson of GoldFire Studios
 *  goldfirestudios.com
 *
 *  MIT License
 */(function(){HowlerGlobal.prototype._pos=[0,0,0],HowlerGlobal.prototype._orientation=[0,0,-1,0,1,0],HowlerGlobal.prototype.stereo=function(t){var r=this;if(!r.ctx||!r.ctx.listener)return r;for(var a=r._howls.length-1;a>=0;a--)r._howls[a].stereo(t);return r},HowlerGlobal.prototype.pos=function(t,r,a){var l=this;if(!l.ctx||!l.ctx.listener)return l;if(r=typeof r!="number"?l._pos[1]:r,a=typeof a!="number"?l._pos[2]:a,typeof t=="number")l._pos=[t,r,a],typeof l.ctx.listener.positionX<"u"?(l.ctx.listener.positionX.setTargetAtTime(l._pos[0],Howler.ctx.currentTime,.1),l.ctx.listener.positionY.setTargetAtTime(l._pos[1],Howler.ctx.currentTime,.1),l.ctx.listener.positionZ.setTargetAtTime(l._pos[2],Howler.ctx.currentTime,.1)):l.ctx.listener.setPosition(l._pos[0],l._pos[1],l._pos[2]);else return l._pos;return l},HowlerGlobal.prototype.orientation=function(t,r,a,l,d,m){var g=this;if(!g.ctx||!g.ctx.listener)return g;var _=g._orientation;if(r=typeof r!="number"?_[1]:r,a=typeof a!="number"?_[2]:a,l=typeof l!="number"?_[3]:l,d=typeof d!="number"?_[4]:d,m=typeof m!="number"?_[5]:m,typeof t=="number")g._orientation=[t,r,a,l,d,m],typeof g.ctx.listener.forwardX<"u"?(g.ctx.listener.forwardX.setTargetAtTime(t,Howler.ctx.currentTime,.1),g.ctx.listener.forwardY.setTargetAtTime(r,Howler.ctx.currentTime,.1),g.ctx.listener.forwardZ.setTargetAtTime(a,Howler.ctx.currentTime,.1),g.ctx.listener.upX.setTargetAtTime(l,Howler.ctx.currentTime,.1),g.ctx.listener.upY.setTargetAtTime(d,Howler.ctx.currentTime,.1),g.ctx.listener.upZ.setTargetAtTime(m,Howler.ctx.currentTime,.1)):g.ctx.listener.setOrientation(t,r,a,l,d,m);else return _;return g},Howl.prototype.init=(function(t){return function(r){var a=this;return a._orientation=r.orientation||[1,0,0],a._stereo=r.stereo||null,a._pos=r.pos||null,a._pannerAttr={coneInnerAngle:typeof r.coneInnerAngle<"u"?r.coneInnerAngle:360,coneOuterAngle:typeof r.coneOuterAngle<"u"?r.coneOuterAngle:360,coneOuterGain:typeof r.coneOuterGain<"u"?r.coneOuterGain:0,distanceModel:typeof r.distanceModel<"u"?r.distanceModel:"inverse",maxDistance:typeof r.maxDistance<"u"?r.maxDistance:1e4,panningModel:typeof r.panningModel<"u"?r.panningModel:"HRTF",refDistance:typeof r.refDistance<"u"?r.refDistance:1,rolloffFactor:typeof r.rolloffFactor<"u"?r.rolloffFactor:1},a._onstereo=r.onstereo?[{fn:r.onstereo}]:[],a._onpos=r.onpos?[{fn:r.onpos}]:[],a._onorientation=r.onorientation?[{fn:r.onorientation}]:[],t.call(this,r)}})(Howl.prototype.init),Howl.prototype.stereo=function(t,r){var a=this;if(!a._webAudio)return a;if(a._state!=="loaded")return a._queue.push({event:"stereo",action:function(){a.stereo(t,r)}}),a;var l=typeof Howler.ctx.createStereoPanner>"u"?"spatial":"stereo";if(typeof r>"u")if(typeof t=="number")a._stereo=t,a._pos=[t,0,0];else return a._stereo;for(var d=a._getSoundIds(r),m=0;m<d.length;m++){var g=a._soundById(d[m]);if(g)if(typeof t=="number")g._stereo=t,g._pos=[t,0,0],g._node&&(g._pannerAttr.panningModel="equalpower",(!g._panner||!g._panner.pan)&&e(g,l),l==="spatial"?typeof g._panner.positionX<"u"?(g._panner.positionX.setValueAtTime(t,Howler.ctx.currentTime),g._panner.positionY.setValueAtTime(0,Howler.ctx.currentTime),g._panner.positionZ.setValueAtTime(0,Howler.ctx.currentTime)):g._panner.setPosition(t,0,0):g._panner.pan.setValueAtTime(t,Howler.ctx.currentTime)),a._emit("stereo",g._id);else return g._stereo}return a},Howl.prototype.pos=function(t,r,a,l){var d=this;if(!d._webAudio)return d;if(d._state!=="loaded")return d._queue.push({event:"pos",action:function(){d.pos(t,r,a,l)}}),d;if(r=typeof r!="number"?0:r,a=typeof a!="number"?-.5:a,typeof l>"u")if(typeof t=="number")d._pos=[t,r,a];else return d._pos;for(var m=d._getSoundIds(l),g=0;g<m.length;g++){var _=d._soundById(m[g]);if(_)if(typeof t=="number")_._pos=[t,r,a],_._node&&((!_._panner||_._panner.pan)&&e(_,"spatial"),typeof _._panner.positionX<"u"?(_._panner.positionX.setValueAtTime(t,Howler.ctx.currentTime),_._panner.positionY.setValueAtTime(r,Howler.ctx.currentTime),_._panner.positionZ.setValueAtTime(a,Howler.ctx.currentTime)):_._panner.setPosition(t,r,a)),d._emit("pos",_._id);else return _._pos}return d},Howl.prototype.orientation=function(t,r,a,l){var d=this;if(!d._webAudio)return d;if(d._state!=="loaded")return d._queue.push({event:"orientation",action:function(){d.orientation(t,r,a,l)}}),d;if(r=typeof r!="number"?d._orientation[1]:r,a=typeof a!="number"?d._orientation[2]:a,typeof l>"u")if(typeof t=="number")d._orientation=[t,r,a];else return d._orientation;for(var m=d._getSoundIds(l),g=0;g<m.length;g++){var _=d._soundById(m[g]);if(_)if(typeof t=="number")_._orientation=[t,r,a],_._node&&(_._panner||(_._pos||(_._pos=d._pos||[0,0,-.5]),e(_,"spatial")),typeof _._panner.orientationX<"u"?(_._panner.orientationX.setValueAtTime(t,Howler.ctx.currentTime),_._panner.orientationY.setValueAtTime(r,Howler.ctx.currentTime),_._panner.orientationZ.setValueAtTime(a,Howler.ctx.currentTime)):_._panner.setOrientation(t,r,a)),d._emit("orientation",_._id);else return _._orientation}return d},Howl.prototype.pannerAttr=function(){var t=this,r=arguments,a,l,d;if(!t._webAudio)return t;if(r.length===0)return t._pannerAttr;if(r.length===1)if(typeof r[0]=="object")a=r[0],typeof l>"u"&&(a.pannerAttr||(a.pannerAttr={coneInnerAngle:a.coneInnerAngle,coneOuterAngle:a.coneOuterAngle,coneOuterGain:a.coneOuterGain,distanceModel:a.distanceModel,maxDistance:a.maxDistance,refDistance:a.refDistance,rolloffFactor:a.rolloffFactor,panningModel:a.panningModel}),t._pannerAttr={coneInnerAngle:typeof a.pannerAttr.coneInnerAngle<"u"?a.pannerAttr.coneInnerAngle:t._coneInnerAngle,coneOuterAngle:typeof a.pannerAttr.coneOuterAngle<"u"?a.pannerAttr.coneOuterAngle:t._coneOuterAngle,coneOuterGain:typeof a.pannerAttr.coneOuterGain<"u"?a.pannerAttr.coneOuterGain:t._coneOuterGain,distanceModel:typeof a.pannerAttr.distanceModel<"u"?a.pannerAttr.distanceModel:t._distanceModel,maxDistance:typeof a.pannerAttr.maxDistance<"u"?a.pannerAttr.maxDistance:t._maxDistance,refDistance:typeof a.pannerAttr.refDistance<"u"?a.pannerAttr.refDistance:t._refDistance,rolloffFactor:typeof a.pannerAttr.rolloffFactor<"u"?a.pannerAttr.rolloffFactor:t._rolloffFactor,panningModel:typeof a.pannerAttr.panningModel<"u"?a.pannerAttr.panningModel:t._panningModel});else return d=t._soundById(parseInt(r[0],10)),d?d._pannerAttr:t._pannerAttr;else r.length===2&&(a=r[0],l=parseInt(r[1],10));for(var m=t._getSoundIds(l),g=0;g<m.length;g++)if(d=t._soundById(m[g]),d){var _=d._pannerAttr;_={coneInnerAngle:typeof a.coneInnerAngle<"u"?a.coneInnerAngle:_.coneInnerAngle,coneOuterAngle:typeof a.coneOuterAngle<"u"?a.coneOuterAngle:_.coneOuterAngle,coneOuterGain:typeof a.coneOuterGain<"u"?a.coneOuterGain:_.coneOuterGain,distanceModel:typeof a.distanceModel<"u"?a.distanceModel:_.distanceModel,maxDistance:typeof a.maxDistance<"u"?a.maxDistance:_.maxDistance,refDistance:typeof a.refDistance<"u"?a.refDistance:_.refDistance,rolloffFactor:typeof a.rolloffFactor<"u"?a.rolloffFactor:_.rolloffFactor,panningModel:typeof a.panningModel<"u"?a.panningModel:_.panningModel};var M=d._panner;M||(d._pos||(d._pos=t._pos||[0,0,-.5]),e(d,"spatial"),M=d._panner),M.coneInnerAngle=_.coneInnerAngle,M.coneOuterAngle=_.coneOuterAngle,M.coneOuterGain=_.coneOuterGain,M.distanceModel=_.distanceModel,M.maxDistance=_.maxDistance,M.refDistance=_.refDistance,M.rolloffFactor=_.rolloffFactor,M.panningModel=_.panningModel}return t},Sound.prototype.init=(function(t){return function(){var r=this,a=r._parent;r._orientation=a._orientation,r._stereo=a._stereo,r._pos=a._pos,r._pannerAttr=a._pannerAttr,t.call(this),r._stereo?a.stereo(r._stereo):r._pos&&a.pos(r._pos[0],r._pos[1],r._pos[2],r._id)}})(Sound.prototype.init),Sound.prototype.reset=(function(t){return function(){var r=this,a=r._parent;return r._orientation=a._orientation,r._stereo=a._stereo,r._pos=a._pos,r._pannerAttr=a._pannerAttr,r._stereo?a.stereo(r._stereo):r._pos?a.pos(r._pos[0],r._pos[1],r._pos[2],r._id):r._panner&&(r._panner.disconnect(0),r._panner=void 0,a._refreshBuffer(r)),t.call(this)}})(Sound.prototype.reset);var e=function(t,r){r=r||"spatial",r==="spatial"?(t._panner=Howler.ctx.createPanner(),t._panner.coneInnerAngle=t._pannerAttr.coneInnerAngle,t._panner.coneOuterAngle=t._pannerAttr.coneOuterAngle,t._panner.coneOuterGain=t._pannerAttr.coneOuterGain,t._panner.distanceModel=t._pannerAttr.distanceModel,t._panner.maxDistance=t._pannerAttr.maxDistance,t._panner.refDistance=t._pannerAttr.refDistance,t._panner.rolloffFactor=t._pannerAttr.rolloffFactor,t._panner.panningModel=t._pannerAttr.panningModel,typeof t._panner.positionX<"u"?(t._panner.positionX.setValueAtTime(t._pos[0],Howler.ctx.currentTime),t._panner.positionY.setValueAtTime(t._pos[1],Howler.ctx.currentTime),t._panner.positionZ.setValueAtTime(t._pos[2],Howler.ctx.currentTime)):t._panner.setPosition(t._pos[0],t._pos[1],t._pos[2]),typeof t._panner.orientationX<"u"?(t._panner.orientationX.setValueAtTime(t._orientation[0],Howler.ctx.currentTime),t._panner.orientationY.setValueAtTime(t._orientation[1],Howler.ctx.currentTime),t._panner.orientationZ.setValueAtTime(t._orientation[2],Howler.ctx.currentTime)):t._panner.setOrientation(t._orientation[0],t._orientation[1],t._orientation[2])):(t._panner=Howler.ctx.createStereoPanner(),t._panner.pan.setValueAtTime(t._stereo,Howler.ctx.currentTime)),t._panner.connect(t._node),t._paused||t._parent.pause(t._id,!0).play(t._id,!0)}})()})(yf)),yf}var IT=DT();const NT=(s,e=!1)=>{const[t,r]=it.useState(!1),[a,l]=it.useState(!1),d=it.useRef(null);it.useEffect(()=>{let _=!1;const M=new IT.Howl({src:[s],loop:!0,volume:.5,html5:!0,preload:!1,onloaderror:()=>l(!0),onload:()=>{e&&!_&&!M.playing()&&M.play()},onplay:()=>{r(!0),l(!1)},onpause:()=>r(!1),onstop:()=>r(!1),onplayerror:()=>{l(!0),r(!1),M.once("unlock",()=>{e&&!_&&M.play()})}});return d.current=M,()=>{_=!0,M.unload(),d.current=null}},[s,e]);const m=it.useCallback(()=>{const _=d.current;_&&(_.playing()?_.pause():_.play())},[]),g=it.useCallback(_=>{d.current&&d.current.volume(_)},[]);return{isPlaying:t,togglePlay:m,setVolume:g,error:a}},UT=[{id:"heart",zh:"爱心",en:"Heart"},{id:"flower",zh:"花朵",en:"Flower"},{id:"saturn",zh:"土星",en:"Saturn"},{id:"fireworks",zh:"烟花",en:"Fireworks"}];function FT(){const[s,e]=it.useState(()=>{try{return localStorage.getItem("site-language")==="en"}catch{return!1}}),t=(A,P)=>s?P:A,[r,a]=it.useState(!1),l=LT(r),[d,m]=it.useState("heart"),[g,_]=it.useState("fist"),[M,u]=it.useState(0),[f,p]=it.useState("#f6c76b"),y=NT("./assets/bgm.mp4",!1),E=r&&l.isReady&&!l.error?l.gesture:g,S=r&&l.isReady&&!l.error?l.handRotation:M;function v(){const A=s?"zh-CN":"en";e(!s),document.documentElement.lang=A;try{localStorage.setItem("site-language",A)}catch{}}return ut.jsxs("div",{className:"particle-stage",lang:s?"en":"zh-CN",children:[ut.jsx(gT,{gesture:E,handRotation:S,model:d,themeColor:f}),ut.jsxs("header",{className:"stage-header",children:[ut.jsxs("div",{children:[ut.jsxs("a",{className:"back-link",href:"../",children:["← ",t("互动实验室","Playground")]}),ut.jsx("h1",{children:"Particle Animation"})]}),ut.jsx("button",{type:"button",onClick:v,"aria-label":"Switch language / 切换语言",children:s?"中文":"EN"})]}),ut.jsxs("section",{className:"stage-controls","aria-label":t("粒子控制","Particle controls"),children:[ut.jsxs("div",{className:"control-heading",children:[ut.jsx("span",{children:"STARLIT PARTICLE STAGE"}),ut.jsx("span",{className:"live-dot","aria-hidden":"true"})]}),ut.jsx("div",{className:"model-picker",children:UT.map(A=>ut.jsx("button",{type:"button","aria-pressed":d===A.id,onClick:()=>{m(A.id),_("fist")},children:s?A.en:A.zh},A.id))}),ut.jsxs("div",{className:"gesture-picker",children:[ut.jsx("button",{type:"button","aria-pressed":E==="fist",onClick:()=>{a(!1),_("fist")},children:t("聚合","Gather")}),ut.jsx("button",{type:"button","aria-pressed":E==="open",onClick:()=>{a(!1),_("open")},children:t("散开","Scatter")}),ut.jsxs("label",{className:"color-control",children:[t("颜色","Color"),ut.jsx("input",{"aria-label":t("粒子颜色","Particle color"),type:"color",value:f,onChange:A=>p(A.target.value)})]})]}),ut.jsxs("label",{className:"rotation-control",children:[t("旋转","Rotation"),ut.jsx("input",{type:"range",min:"-1",max:"1",step:"0.01",value:M,onChange:A=>{a(!1),u(Number(A.target.value))}})]}),ut.jsxs("div",{className:"extra-controls",children:[ut.jsx("button",{type:"button","aria-pressed":r,onClick:()=>a(!r),children:r?t("关闭摄像头","Stop camera"):t("开启手势控制","Enable gestures")}),ut.jsx("button",{type:"button","aria-pressed":y.isPlaying,onClick:y.togglePlay,children:y.isPlaying?t("暂停音乐","Pause music"):t("播放音乐","Play music")})]}),ut.jsx("p",{className:"camera-status",role:"status",children:l.error?t("摄像头或手势模型暂不可用，请使用上方控制按钮。可关闭后重试。","Camera or gesture model unavailable. Use the controls above, or turn the camera off and retry."):r?l.isReady?t("握拳聚合 · 张手散开 · 转动手腕旋转","Fist to gather · Open hand to scatter · Turn wrist to rotate"):t("正在准备摄像头与手势模型…","Preparing camera and gesture model…"):t("无需摄像头也可体验；开启手势后，画面仅在本机处理。","No camera needed. Gesture video is processed on your device.")}),y.error&&ut.jsx("p",{role:"status",className:"camera-status",children:t("音乐暂时无法播放，请再次点击播放。","Audio unavailable. Click play to retry.")}),ut.jsxs("details",{children:[ut.jsx("summary",{children:t("关于这个小实验","About this experiment")}),ut.jsx("p",{children:t("探索爱心、花朵、土星与烟花四种粒子造型。切换造型后等待片刻，让粒子慢慢聚拢。手势首次开启需要下载识别模型。","Explore hearts, flowers, Saturn and fireworks. Allow a moment for each transition. Gesture recognition downloads its model on first use.")})]})]}),ut.jsx("aside",{className:`camera-preview ${r?"camera-visible":""}`,"aria-label":t("摄像头预览","Camera preview"),children:ut.jsx("video",{ref:l.videoRef,playsInline:!0,muted:!0,autoPlay:!0})}),ut.jsx("img",{className:"stage-keepsake",src:"./assets/jf_2.jpg",alt:"Sherlock Holmes"})]})}D0.createRoot(document.getElementById("root")).render(ut.jsx(it.StrictMode,{children:ut.jsx(FT,{})}));export{Vo as c,OT as g};
