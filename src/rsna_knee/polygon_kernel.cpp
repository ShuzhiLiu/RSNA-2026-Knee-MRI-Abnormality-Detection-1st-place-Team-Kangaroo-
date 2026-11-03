#include <vector>
#include <cmath>
#include <algorithm>
struct P {double x,y;};
using Poly=std::vector<P>;
double cross(P d,P q,P a){return d.x*(q.y-a.y)-d.y*(q.x-a.x);}
double signed2(const Poly&p){double a=0,b=0; for(size_t i=0;i<p.size();++i){size_t j=(i+1)%p.size();a+=p[i].x*p[j].y;b+=p[i].y*p[j].x;} return a-b;}
Poly clip(const Poly&pp,const Poly&bound){Poly p=pp;double ss=signed2(bound),sign=(ss>0)-(ss<0);for(size_t i=0;i<bound.size();++i){if(p.empty())break;P a=bound[i],b=bound[(i+1)%bound.size()],d={b.x-a.x,b.y-a.y};Poly out;for(size_t k=0;k<p.size();++k){P u=p[k],v=p[(k+1)%p.size()];double cu=sign*cross(d,u,a),cv=sign*cross(d,v,a);bool inu=cu>=-1e-9,inv=cv>=-1e-9;if(inu)out.push_back(u);if(inu!=inv){double f=cu/(cu-cv);out.push_back({u.x+(v.x-u.x)*f,u.y+(v.y-u.y)*f});}}p=std::move(out);}return p;}
extern "C" void overlap(const double*points,const int*lengths,int n,const double*vp,int vn,int G,int size,float*out){Poly valid;for(int i=0;i<vn;i++)valid.push_back({vp[2*i],vp[2*i+1]});std::vector<double> a(G*G,0.);double s=double(size)/G;int offset=0;for(int k=0;k<n;k++){Poly p;for(int i=0;i<lengths[k];i++)p.push_back({points[2*(offset+i)],points[2*(offset+i)+1]});offset+=lengths[k];p=clip(p,valid);if(p.size()<3)continue;double x0=p[0].x,x1=x0,y0=p[0].y,y1=y0;for(auto q:p){x0=std::min(x0,q.x);x1=std::max(x1,q.x);y0=std::min(y0,q.y);y1=std::max(y1,q.y);}for(int y=std::max(0,int(std::floor(y0/s)));y<=std::min(G-1,int(std::floor(y1/s)));++y)for(int x=std::max(0,int(std::floor(x0/s)));x<=std::min(G-1,int(std::floor(x1/s)));++x){Poly cell={{x*s,y*s},{(x+1)*s,y*s},{(x+1)*s,(y+1)*s},{x*s,(y+1)*s}};Poly c=clip(p,cell);double ar=c.size()>2?std::abs(signed2(c))*.5:0.;a[y*G+x]+=ar/(s*s);}}for(int i=0;i<G*G;i++)out[i]=float(std::min(a[i],1.));}
