## Corey Chivers, 2012 ##
sim_bayes<-function(p=0.5,N=100,y_lim=20,a_a=2,a_b=10,b_a=8,b_b=3)
{
  ## Simulate outcomes in advance   
  outcomes<-sample(1:0,N,prob=c(p,1-p),replace=TRUE)
  success<-cumsum(outcomes)
  
  for(frame inmr  1:N)
  {
    #png(paste("/sandbox/rnd/plots/",1000+frame,".png",sep=""))
    curve(dbeta(x,a_a,a_b),xlim=c(0,1),ylim=c(0,y_lim),col='green',xlab='p',ylab='Posterior Density',lty=2)
    curve(dbeta(x,b_a,b_b),col='blue',lty=2,add=TRUE)
    #plot(x, dbeta(x,a_a,a_b),xlim=c(0,1),ylim=c(0,y_lim),col='green',xlab='p',ylab='Posterior Density',lty=2)
    #plot(x, dbeta(x,b_a,b_b),col='blue',lty=2,add=TRUE)
    
    for(i in 1:frame)
    {
      curve(dbeta(x,a_a+success[i]+1,a_b+(i-success[i])+1),add=TRUE,col=rgb(0,100,0,(1-(frame-i)/frame) * 100,maxColorValue=255))
      curve(dbeta(x,b_a+success[i]+1,b_b+(i-success[i])+1),add=TRUE,col=rgb(0,0,100,(1-(frame-i)/frame) * 100,maxColorValue=255))
      #plot(x, dbeta(x,a_a+success[i]+1,a_b+(i-success[i])+1),add=TRUE,col=rgb(0,100,0,(1-(frame-i)/frame) * 100,maxColorValue=255))
      #plot(x, dbeta(x,b_a+success[i]+1,b_b+(i-success[i])+1),add=TRUE,col=rgb(0,0,100,(1-(frame-i)/frame) * 100,maxColorValue=255))
    }
    curve(dbeta(x,a_a+success[i]+1,a_b+(i-success[i])+1),add=TRUE,col=rgb(0,100,0,255,maxColorValue=255),lwd=2)
    curve(dbeta(x,b_a+success[i]+1,b_b+(i-success[i])+1),add=TRUE,col=rgb(0,0,100,255,maxColorValue=255),lwd=2)
    #plot(x, dbeta(x,a_a+success[i]+1,a_b+(i-success[i])+1),add=TRUE,col=rgb(0,100,0,255,maxColorValue=255),lwd=2)
    #plot(x, dbeta(x,b_a+success[i]+1,b_b+(i-success[i])+1),add=TRUE,col=rgb(0,0,100,255,maxColorValue=255),lwd=2)
    
    legend('topleft',legend=c('Observer A','Observer B'),lty=1,col=c('green','blue'))
    text(0.75,17,label=paste(success[i],"successes,",i-success[i],"failures"))
    #dev.off()
  }
}