"use strict";
/* The network demo for the network lesson: a small network trained live on a 3D double helix. Training runs in a Web Worker. */
function hxSimpleStep(api){
  var vp = api.el('pn', 'left:28px;top:20px;width:904px;height:500px;padding:0', '<canvas width="1800" height="1000" style="width:900px;height:500px"></canvas>');
  function opt(vals, sel){ return vals.map(function(v){ return '<option' + (String(v) === String(sel) ? ' selected' : '') + '>' + v + '</option>'; }).join(''); }
  var els = {};
  function ctl(label, cls, html){ var w = api.addControl(label + ' ' + html); els[cls] = qs('.' + cls, w); }
  ctl('Training points', 'npt', '<select class="npt">' + opt([10, 100, 1000, 10000, 100000], 10000) + '</select>');
  ctl('Neurons per layer', 'wid', '<select class="wid">' + opt([4, 8, 16, 32], 32) + '</select>');
  ctl('Hidden layers', 'lay', '<select class="lay">' + opt([1, 2, 3, 4], 3) + '</select>');
  ctl('Activation', 'act', '<select class="act"><option value="relu" selected>ReLU</option><option value="tanh">tanh</option></select>');
  ctl('Learning rate', 'lr', '<select class="lr">' + opt([0.001, 0.003, 0.01, 0.03], 0.003) + '</select>');
  ctl('Rotate', 'rot', '<input class="rot" type="checkbox" checked>');
  ctl('Slice height', 'sl', '<input class="sl" type="range" min="-80" max="80" value="0" style="width:110px">');
  var bw = api.addControl('<button type="button" class="btn sm play on">Pause</button> <button type="button" class="btn sm rst">Restart</button>'); els.play = qs('.play', bw); els.rst = qs('.rst', bw);
function WSRC(){
 var R=0.45,RS=0.2,K=2*Math.PI*1.5/1.7,ZM=0.85,RR=0.065;
 function sdf(x,y,z){var th=Math.atan2(y,x),best=9;
  for(var s=0;s<2;s++){var ph=s*Math.PI;for(var n=-3;n<=3;n++){var zz=(th-ph+2*Math.PI*n)/K;if(zz<-ZM-0.3||zz>ZM+0.3)continue;zz=zz<-ZM?-ZM:zz>ZM?ZM:zz;var dx=x-R*Math.cos(K*zz+ph),dy=y-R*Math.sin(K*zz+ph),dz=z-zz,d=Math.sqrt(dx*dx+dy*dy+dz*dz)-RS;if(d<best)best=d;}}
  for(var j=0;j<5;j++){var zj=-0.6+0.3*j;if(Math.abs(z-zj)>0.25)continue;var ax=R*Math.cos(K*zj),ay=R*Math.sin(K*zj),t=(x*ax+y*ay)/(ax*ax+ay*ay);t=t<-1?-1:t>1?1:t;var px=x-t*ax,py=y-t*ay,pz=z-zj,d2=Math.sqrt(px*px+py*py+pz*pz)-RR;if(d2<best)best=d2;}return best;}
 function rnd(s){var a=s;return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
 var P,net,adam,data,test,step=0,playing=false,total=30000,timer=null,GA,DL,GW,GB,gen=0,hold=false,pending=false;
 function band(r){for(var k=0;k<400;k++){var a=r()*6.2832,rr=0.8*Math.sqrt(r()),x=rr*Math.cos(a),y=rr*Math.sin(a),z=(r()*2-1)*1.0;if(Math.abs(sdf(x,y,z))<0.07)return [x,y,z];}return [0,0,0];}
 function mkdata(){var r=rnd(7+P.seed*13+P.N),N=P.N,X=new Float32Array(N*3),Y=new Float32Array(N);
  for(var i=0;i<N;i++){var u=r(),x,y,z;if(u<0.3){x=r()*2-1;y=r()*2-1;z=r()*2-1;}else if(u<0.5){var a=r()*6.2832,rr=0.85*Math.sqrt(r());x=rr*Math.cos(a);y=rr*Math.sin(a);z=(r()*2-1)*0.95;}else{var q=band(r);x=q[0];y=q[1];z=q[2];}
   X[3*i]=x;X[3*i+1]=y;X[3*i+2]=z;Y[i]=sdf(x,y,z)<0?1:-1;}data={X:X,Y:Y};
  var r2=rnd(999),T=3000;test={X:new Float32Array(T*3),Y:new Float32Array(T),n:T};for(var j=0;j<T;j++){var q2=band(r2);test.X[3*j]=q2[0];test.X[3*j+1]=q2[1];test.X[3*j+2]=q2[2];test.Y[j]=sdf(q2[0],q2[1],q2[2])<0?1:-1;}}
 function init(){var S=[3];for(var i=0;i<P.L;i++)S.push(P.W);S.push(1);var r=rnd(5+P.W+P.seed*31),nl=S.length-1;net={S:S,W:[],b:[]};adam={m:[],v:[],mb:[],vb:[],t:0};
  for(var l=0;l<nl;l++){var w=new Float32Array(S[l]*S[l+1]),sc=P.act==='relu'&&l<nl-1?Math.sqrt(2/S[l]):Math.sqrt(6/(S[l]+S[l+1]));if(l===0)sc*=P.act==='relu'?2.2:3;for(var k=0;k<w.length;k++)w[k]=(r()*2-1)*sc*(P.act==='relu'?1.7:1);net.W.push(w);net.b.push(new Float32Array(S[l+1]));adam.m.push(new Float32Array(w.length));adam.v.push(new Float32Array(w.length));adam.mb.push(new Float32Array(S[l+1]));adam.vb.push(new Float32Array(S[l+1]));}
  for(var q=0;q<S[1];q++)net.b[0][q]=(r()*2-1)*1.2;step=0;GA=mkA();DL=mkA();}
 function mkA(){return net.S.map(function(n){return new Float32Array(n);});}
 function fwd(x,y,z,A){var S=net.S,nl=S.length-1,relu=P.act==='relu';A[0][0]=x;A[0][1]=y;A[0][2]=z;for(var l=0;l<nl;l++){var ni=S[l],no=S[l+1],w=net.W[l],b=net.b[l],a=A[l],o=A[l+1],last=l===nl-1;for(var j=0;j<no;j++){var s=b[j];for(var i=0;i<ni;i++)s+=a[i]*w[i*no+j];o[j]=last?Math.tanh(s):relu?(s>0?s:0):Math.tanh(s);}}return A[nl][0];}
 function train(lr){var S=net.S,nl=S.length-1,N=P.N,bs=Math.min(128,N),relu=P.act==='relu';GW=net.W.map(function(w){return new Float32Array(w.length);});GB=net.b.map(function(b){return new Float32Array(b.length);});
  for(var q=0;q<bs;q++){var idx=Math.floor(Math.random()*N),p=fwd(data.X[3*idx],data.X[3*idx+1],data.X[3*idx+2],GA);DL[nl][0]=2*(p-data.Y[idx])*(1-p*p)/bs;
   for(var l=nl-1;l>=0;l--){var ni=S[l],no=S[l+1],w=net.W[l],gw=GW[l],gb=GB[l],a=GA[l],d=DL[l+1];for(var j=0;j<no;j++)gb[j]+=d[j];for(var i=0;i<ni;i++){var ai=a[i],s=0;for(var j2=0;j2<no;j2++){gw[i*no+j2]+=ai*d[j2];s+=w[i*no+j2]*d[j2];}if(l>0)DL[l][i]=relu?(ai>0?s:0):s*(1-ai*ai);}}}
  adam.t++;var c1=1-Math.pow(0.9,adam.t),c2=1-Math.pow(0.999,adam.t);
  for(var l2=0;l2<nl;l2++){var w2=net.W[l2],g2=GW[l2],m=adam.m[l2],v=adam.v[l2],k;for(k=0;k<w2.length;k++){var gg=g2[k];m[k]=0.9*m[k]+0.1*gg;v[k]=0.999*v[k]+0.001*gg*gg;w2[k]-=lr*(m[k]/c1)/(Math.sqrt(v[k]/c2)+1e-8);}var b2=net.b[l2],gb2=GB[l2],mb=adam.mb[l2],vb=adam.vb[l2];for(k=0;k<b2.length;k++){var g3=gb2[k];mb[k]=0.9*mb[k]+0.1*g3;vb[k]=0.999*vb[k]+0.001*g3*g3;b2[k]-=lr*(mb[k]/c1)/(Math.sqrt(vb[k]/c2)+1e-8);}}
  step++;}
 function curLR(){var t=Math.min(1,step/total);return P.lr*(0.05+0.95*0.5*(1+Math.cos(Math.PI*t)));}
 function bacc(){var A=mkA(),tp=0,tn=0,p=0,n=0;for(var i=0;i<test.n;i++){var o=fwd(test.X[3*i],test.X[3*i+1],test.X[3*i+2],A)>0;if(test.Y[i]>0){p++;if(o)tp++;}else{n++;if(!o)tn++;}}return 0.5*(tp/Math.max(1,p)+tn/Math.max(1,n));}
 function field(G){var A=mkA(),f=new Float32Array(G*G*G);for(var k=0;k<G;k++)for(var j=0;j<G;j++)for(var i=0;i<G;i++)f[(k*G+j)*G+i]=fwd((i+0.5)/G*2-1,(j+0.5)/G*2-1,(k+0.5)/G*2-1,A);return f;}
 function post(final){var G=final?72:40,f=field(G),W=net.W.map(function(w){return new Float32Array(w);}),B=net.b.map(function(b){return new Float32Array(b);}),tr=[f.buffer];W.forEach(function(w){tr.push(w.buffer);});B.forEach(function(b){tr.push(b.buffer);});
  pending=true;self.postMessage({type:'snap',step:step,acc:bacc(),G:G,field:f,W:W,b:B,S:net.S,final:!!final,gen:gen},tr);}
 function loop(){if(!playing)return;if(hold||pending){timer=setTimeout(loop,80);return;}if(step>=total){playing=false;post(true);return;}
  var t0=Date.now(),chunk=Math.max(4,Math.round(step*0.4)),budget=step<800?1e9:260;
  for(var i=0;i<chunk&&step<total;i++){train(curLR());if(Date.now()-t0>budget)break;}
  post(false);var delay=step<700?260:step<2000?120:80;timer=setTimeout(loop,delay);}
 self.onmessage=function(e){var m=e.data;if(m.cmd==='init'){playing=false;clearTimeout(timer);gen=m.gen;pending=false;P=m.P;total=m.total||30000;mkdata();init();post(false);playing=true;timer=setTimeout(loop,200);}
  else if(m.cmd==='ack'){pending=false;}else if(m.cmd==='hold'){hold=true;}else if(m.cmd==='unhold'){hold=false;}else if(m.cmd==='pause'){playing=false;clearTimeout(timer);post(true);}else if(m.cmd==='play'){if(!playing){playing=true;timer=setTimeout(loop,10);}}};
}
var blob=new Blob(['('+WSRC.toString()+')()'],{type:'text/javascript'}),worker=null;
var CW=900,CH=500,cv=qs('canvas',vp),dpr=2;cv.width=CW*dpr;cv.height=CH*dpr;
var g=cv.getContext('2d');g.scale(dpr,dpr);
var TXT='#2a2c33',TX2='#6a6d76',BRD='#b9bcc4',FONT='12px Aptos,"Segoe UI",sans-serif';
var R=0.45,RS=0.2,K=2*Math.PI*1.5/1.7,ZM=0.85,RR=0.065;
function sdf(x,y,z){var th=Math.atan2(y,x),best=9;
 for(var s=0;s<2;s++){var ph=s*Math.PI;for(var n=-3;n<=3;n++){var zz=(th-ph+2*Math.PI*n)/K;if(zz<-ZM-0.3||zz>ZM+0.3)continue;zz=zz<-ZM?-ZM:zz>ZM?ZM:zz;var dx=x-R*Math.cos(K*zz+ph),dy=y-R*Math.sin(K*zz+ph),dz=z-zz,d=Math.sqrt(dx*dx+dy*dy+dz*dz)-RS;if(d<best)best=d;}}
 for(var j=0;j<5;j++){var zj=-0.6+0.3*j;if(Math.abs(z-zj)>0.25)continue;var ax=R*Math.cos(K*zj),ay=R*Math.sin(K*zj),t=(x*ax+y*ay)/(ax*ax+ay*ay);t=t<-1?-1:t>1?1:t;var px=x-t*ax,py=y-t*ay,pz=z-zj,d2=Math.sqrt(px*px+py*py+pz*pz)-RR;if(d2<best)best=d2;}return best;}
function extract(f,G,flip){var pos=[],nor=[],h=2/G;
 function at(i,j,k){i=i<0?0:i>=G?G-1:i;j=j<0?0:j>=G?G-1:j;k=k<0?0:k>=G?G-1:k;var v=f[(k*G+j)*G+i];return flip?-v:v;}
 function gr(i,j,k){var gx=at(i+1,j,k)-at(i-1,j,k),gy=at(i,j+1,k)-at(i,j-1,k),gz=at(i,j,k+1)-at(i,j,k-1),l=Math.sqrt(gx*gx+gy*gy+gz*gz)||1;return [gx/l,gy/l,gz/l];}
 for(var k=0;k<G;k++)for(var j=0;j<G;j++)for(var i=0;i<G;i++){var a=at(i,j,k);var nb=[[1,0,0],[0,1,0],[0,0,1]];for(var d=0;d<3;d++){var i2=i+nb[d][0],j2=j+nb[d][1],k2=k+nb[d][2];if(i2>=G||j2>=G||k2>=G)continue;var b=at(i2,j2,k2);if((a<0)!==(b<0)){var t=a/(a-b),x=(i+0.5+t*nb[d][0])*h-1,y=(j+0.5+t*nb[d][1])*h-1,z=(k+0.5+t*nb[d][2])*h-1,n=gr(i,j,k);pos.push(x,y,z);nor.push(n[0],n[1],n[2]);}}}
 return {P:new Float32Array(pos),N:new Float32Array(nor),n:pos.length/3,h:h};}
var surfT=null,lSurf=null,snap=null,tImg=null,lImg=null,ang=0.6,tilt=0.32,PS=240,sliceZ=0,play=true;
function makeSurface(){var G=72,f=new Float32Array(G*G*G);for(var k=0;k<G;k++)for(var j=0;j<G;j++)for(var i=0;i<G;i++)f[(k*G+j)*G+i]=sdf((i+0.5)/G*2-1,(j+0.5)/G*2-1,(k+0.5)/G*2-1);return extract(f,G,false);}
var RB=[null,null];
function render3d(surf,base,slot){
 var S2=Math.round(PS*1.6),b=RB[slot];if(!b){var c=document.createElement('canvas');c.width=S2;c.height=S2;var cx0=c.getContext('2d');b=RB[slot]={c:c,x:cx0,im:cx0.createImageData(S2,S2),zb:new Float32Array(S2*S2)};}
 var x=b.x,im=b.im,d=im.data,zb=b.zb;zb.fill(-9);new Uint32Array(d.buffer).fill(0xFFF9F7F6);
 if(!surf){x.putImageData(im,0,0);return b.c;}
 var ca=Math.cos(ang),sa=Math.sin(ang),ct=Math.cos(tilt),st=Math.sin(tilt),sc=S2*0.40,cx=S2/2,cy=S2/2,L=[-0.35,0.45,0.82],P0=surf.P,N0=surf.N,n=surf.n,rad=Math.max(1,Math.round(surf.h*sc*0.85));
 for(var q=0;q<n;q++){var px=P0[3*q],py=P0[3*q+1],pz=P0[3*q+2],x1=px*ca-py*sa,y1=px*sa+py*ca,y2=y1*ct-pz*st,z2=y1*st+pz*ct,nx=N0[3*q],ny=N0[3*q+1],nz=N0[3*q+2],nx1=nx*ca-ny*sa,ny1=nx*sa+ny*ca,ny2=ny1*ct-nz*st,nz2=ny1*st+nz*ct,dep=y2,sxp=Math.round(cx+x1*sc),syp=Math.round(cy-z2*sc);
  var li=0.62+0.50*Math.max(0,nx1*L[0]+nz2*L[2]-ny2*0.25)-0.12*Math.max(0,-nz2),hv=Math.max(0,nz2*0.6-ny2*0.7),sp=Math.pow(hv,18)*0.35,fog=1-0.22*((dep+1)/2),rr=Math.min(255,(base[0]*li+255*sp)*fog),gg=Math.min(255,(base[1]*li+255*sp)*fog),bb=Math.min(255,(base[2]*li+255*sp)*fog);
  for(var oy=-rad;oy<=rad;oy++)for(var ox=-rad;ox<=rad;ox++){if(ox*ox+oy*oy>rad*rad+1)continue;var X=sxp+ox,Y=syp+oy;if(X<0||Y<0||X>=S2||Y>=S2)continue;var id=Y*S2+X,dz=dep+0.004*(rad*rad-ox*ox-oy*oy);if(dz>zb[id]){zb[id]=dz;d[id*4]=rr;d[id*4+1]=gg;d[id*4+2]=bb;}}}
 x.putImageData(im,0,0);return b.c;}
function hex(h){return [parseInt(h.substr(1,2),16),parseInt(h.substr(3,2),16),parseInt(h.substr(5,2),16)];}
var NEG=hex('#378ADD'),POS=hex('#D85A30'),MID=[246,246,242];
function dv(v){var t=Math.max(-1,Math.min(1,v)),c=t<0?NEG:POS,a=Math.abs(t);return [MID[0]+(c[0]-MID[0])*a,MID[1]+(c[1]-MID[1])*a,MID[2]+(c[2]-MID[2])*a];}
function mk(w,h){var c=document.createElement('canvas');c.width=w;c.height=h;return c;}
function mapC(n,fn){var c=mk(n,n),x=c.getContext('2d'),im=x.createImageData(n,n);for(var j=0;j<n;j++)for(var i=0;i<n;i++){var rgb=fn((i+0.5)/n*2-1,(j+0.5)/n*2-1),o=(j*n+i)*4;im.data[o]=rgb[0];im.data[o+1]=rgb[1];im.data[o+2]=rgb[2];im.data[o+3]=255;}x.putImageData(im,0,0);return c;}
var PA={N:10000,W:32,L:3,act:'relu',lr:0.003,seed:1},maps=null,outSlice=null,tgtSlice=null;
function mfwd(x,y,z,A){var S=snap.S,nl=S.length-1,relu=PA.act==='relu';A[0][0]=x;A[0][1]=y;A[0][2]=z;for(var l=0;l<nl;l++){var ni=S[l],no=S[l+1],w=snap.W[l],b=snap.b[l],a=A[l],o=A[l+1],last=l===nl-1;for(var j=0;j<no;j++){var s=b[j];for(var i=0;i<ni;i++)s+=a[i]*w[i*no+j];o[j]=last?Math.tanh(s):relu?(s>0?s:0):Math.tanh(s);}}return A[nl][0];}
var GT=22;
function mkMaps(){var S=snap.S,A=S.map(function(n){return new Float32Array(n);}),vals=S.map(function(n){return new Float32Array(n*GT*GT);}),relu=PA.act==='relu';
 for(var j=0;j<GT;j++)for(var i=0;i<GT;i++){mfwd((i+0.5)/GT*2-1,(j+0.5)/GT*2-1,sliceZ,A);for(var l=0;l<S.length;l++)for(var k=0;k<S[l];k++)vals[l][k*GT*GT+j*GT+i]=A[l][k];}
 maps=S.map(function(n,l){var arr=[];for(var k=0;k<Math.min(n,16);k++){(function(l4,k4){var c=mk(GT,GT),x=c.getContext('2d'),im=x.createImageData(GT,GT);var mx=1;if(relu&&l4>0&&l4<S.length-1){mx=0.001;for(var p0=0;p0<GT*GT;p0++)mx=Math.max(mx,vals[l4][k4*GT*GT+p0]);}for(var p=0;p<GT*GT;p++){var v=vals[l4][k4*GT*GT+p];if(l4===0&&k4===2)v=sliceZ;if(relu&&l4>0&&l4<S.length-1)v=v/mx;var rgb=dv(v);im.data[p*4]=rgb[0];im.data[p*4+1]=rgb[1];im.data[p*4+2]=rgb[2];im.data[p*4+3]=255;}x.putImageData(im,0,0);arr.push(c);})(l,k);}return arr;});
 var A2=S.map(function(n){return new Float32Array(n);});outSlice=mapC(80,function(x,y){var o=mfwd(x,y,sliceZ,A2),t=(o+1)/2;return [244-206*t,244-202*t,240-196*t];});}
function mkTgtSlice(){tgtSlice=mapC(80,function(x,y){return sdf(x,y,sliceZ)<0?[38,42,50]:[244,244,240];});}
function layout(){var xs=[36];var x0=130,x1=330;for(var l=1;l<=PA.L;l++)xs.push(PA.L===1?200:x0+(x1-x0)*(l-1)/(PA.L-1));return xs;}
var accHist=[];
function draw(){
 g.clearRect(0,0,CW,CH);g.font=FONT;if(!snap||!maps){g.fillStyle=TX2;g.fillText('Preparing training data',20,30);return;}
 var S=snap.S,nl=S.length-1,xs=layout(),Y0=52,Y1=436,OX=430,OY=(Y0+Y1)/2;
 g.fillStyle=TX2;g.fillText('Inputs',10,24);g.fillText('Tile = one neuron, at z = '+sliceZ.toFixed(2),170,24);
 function tp(l,k,n){var nn=Math.min(n,16),h=(Y1-Y0)/nn,s=Math.min(h-3,50);return [xs[l],Y0+h*(k+0.5),s];}
 function inpos(k){return [xs[0],OY+(k-1)*74,42];}
 for(var l=0;l<nl;l++){var ni=S[l],no=S[l+1],w=snap.W[l],dI=l===0?3:Math.min(ni,16),dO=Math.min(no,16);
  for(var i=0;i<dI;i++){var a=l===0?inpos(i):tp(l,i,ni),ax=a[0]+a[2]/2,ay=a[1];
   for(var j=0;j<dO;j++){var bx,by;if(l===nl-1){bx=OX;by=OY;}else{var b=tp(l+1,j,no);bx=b[0]-b[2]/2;by=b[1];}
    var v=w[i*no+j],aw=Math.min(0.5,Math.abs(v)*0.18);if(aw<0.025)continue;g.strokeStyle=v>0?'rgba(216,90,48,'+aw+')':'rgba(55,138,221,'+aw+')';g.lineWidth=Math.min(2,0.3+Math.abs(v)*0.55);g.beginPath();g.moveTo(ax,ay);g.lineTo(bx,by);g.stroke();}}}
 var lab=['x','y','z'];
 for(var k=0;k<3;k++){var p=inpos(k);g.drawImage(maps[0][k],p[0]-p[2]/2,p[1]-p[2]/2,p[2],p[2]);g.strokeStyle=BRD;g.lineWidth=0.6;g.strokeRect(p[0]-p[2]/2,p[1]-p[2]/2,p[2],p[2]);g.fillStyle=TX2;g.fillText(lab[k],p[0]-4,p[1]+p[2]/2+13);}
 for(var l2=1;l2<=PA.L;l2++){var n2=S[l2],dn=Math.min(n2,16);g.fillStyle=TX2;g.fillText(n2+' neurons',xs[l2]-26,Y0-10);for(var k2=0;k2<dn;k2++){var p2=tp(l2,k2,n2);g.imageSmoothingEnabled=true;g.drawImage(maps[l2][k2],p2[0]-p2[2]/2,p2[1]-p2[2]/2,p2[2],p2[2]);g.strokeStyle=BRD;g.lineWidth=0.6;g.strokeRect(p2[0]-p2[2]/2,p2[1]-p2[2]/2,p2[2],p2[2]);}
  if(n2>16){g.fillStyle=TX2;g.fillText('+'+(n2-16)+' more',xs[l2]-22,Y1+12);}}
 g.fillStyle='#fff';g.beginPath();g.arc(OX,OY,6,0,6.2832);g.fill();g.strokeStyle=TXT;g.lineWidth=1;g.stroke();g.fillStyle=TX2;g.fillText('output',OX-16,OY+22);
 g.fillStyle=TX2;g.fillText('Target helix',470,24);g.fillText('Learned by the network',690,24);
 if(tImg){g.drawImage(tImg,470,32,PS*0.92,PS*0.92);g.strokeStyle='#e1e3e8';g.strokeRect(470,32,PS*0.92,PS*0.92);}
 if(lImg){g.drawImage(lImg,690,32,PS*0.92,PS*0.92);g.strokeRect(690,32,PS*0.92,PS*0.92);}
 g.fillStyle=TX2;g.fillText('Target cross-section',470,262);g.fillText('Network output, same cross-section',690,262);
 g.drawImage(tgtSlice,470,270,150,150);g.strokeStyle=BRD;g.strokeRect(470,270,150,150);g.drawImage(outSlice,690,270,150,150);g.strokeRect(690,270,150,150);
 var cx=470,cy=456,cw=390,ch=30;g.strokeStyle=BRD;g.lineWidth=0.6;g.strokeRect(cx,cy,cw,ch);g.fillStyle=TX2;g.fillText('Held-out accuracy near the surface (0.5 is chance), log scale in steps',cx,cy-6);
 if(accHist.length>1){g.strokeStyle='#D85A30';g.lineWidth=2;g.beginPath();for(var h2=0;h2<accHist.length;h2++){var hx=cx+Math.log10(1+accHist[h2][0])/Math.log10(1+12000)*cw,hy=cy+ch-(Math.max(0.5,accHist[h2][1])-0.5)*2*ch;h2?g.lineTo(hx,hy):g.moveTo(hx,hy);}g.stroke();}
 var np=0;for(var q=0;q<nl;q++)np+=snap.W[q].length+snap.b[q].length;
 g.fillStyle=TXT;g.fillText('step '+snap.step.toLocaleString()+'    accuracy '+snap.acc.toFixed(2)+'    parameters '+np.toLocaleString()+'    training points '+PA.N.toLocaleString(),10,CH-6);}
var curGen=0;
function onSnap(m){if(m.gen!==curGen)return;snap=m;accHist.push([m.step,m.acc]);lSurf=extract(m.field,m.G,true);mkMaps();vDirty=true;need=true;if(worker)worker.postMessage({cmd:'ack'});}
function start(){if(worker)worker.terminate();worker=new Worker(URL.createObjectURL(blob));worker.onmessage=function(e){if(e.data.type==='snap')onSnap(e.data);};snap=null;maps=null;accHist=[];lImg=null;lSurf=null;curGen++;mkTgtSlice();vDirty=true;need=true;
 worker.postMessage({cmd:'init',P:PA,total:12000,gen:curGen});if(!live)worker.postMessage({cmd:'hold'});}
var lastRot=0,vis=true,live=true,need=true,vDirty=true,rotOn=true,cost=0;
function isLive(){return vis&&!document.hidden&&!!cv.offsetParent;}
function loop(ts){
 var lv=isLive();if(lv!==live){live=lv;if(worker)worker.postMessage({cmd:lv?'unhold':'hold'});if(lv){need=true;lastRot=ts;}}
 if(!live){setTimeout(function(){requestAnimationFrame(loop);},250);return;}
 var rot=rotOn&&ts-lastRot>Math.max(50,cost*3);
 if(rot){ang+=0.00048*Math.min(200,ts-lastRot);lastRot=ts;}
 if(rot||vDirty){var t0=performance.now();tImg=render3d(surfT,[120,132,150],0);if(lSurf)lImg=render3d(lSurf,[216,100,58],1);var c=performance.now()-t0;cost=cost?cost*0.7+c*0.3:c;vDirty=false;need=true;
  if(cost>45&&rotOn){rotOn=false;if(els.rot)els.rot.checked=false;}}
 if(need){need=false;draw();}
 requestAnimationFrame(loop);}
function ui(){PA.N=+els.npt.value;PA.W=+els.wid.value;PA.L=+els.lay.value;PA.act=els.act.value;PA.lr=+els.lr.value;PA.seed=1;play=true;els.play.textContent='Pause';els.play.className='on';start();}
['npt','wid','lay','act','lr'].forEach(function(id){els[id].addEventListener('change',ui);});
els.rot.addEventListener('change',function(){rotOn=this.checked;cost=0;need=true;});
els.sl.addEventListener('input',function(){sliceZ=this.value/100;mkTgtSlice();if(snap)mkMaps();need=true;});
els.rst.addEventListener('click',function(){PA.seed++;start();});
els.play.addEventListener('click',function(){play=!play;els.play.textContent=play?'Pause':'Play';els.play.className=play?'on':'';worker.postMessage({cmd:play?'play':'pause'});});
var started=false;
return {render:function(){if(started)return;started=true;surfT=makeSurface();if('IntersectionObserver' in window)new IntersectionObserver(function(es){vis=es[es.length-1].isIntersecting;}).observe(cv);ui();requestAnimationFrame(loop);},info:function(){return snap?{step:snap.step,acc:snap.acc,final:snap.final}:null;},set:function(n,w,l,act,lr){els.npt.value=n;els.wid.value=w;els.lay.value=l;els.act.value=act;els.lr.value=lr;ui();}};
}
