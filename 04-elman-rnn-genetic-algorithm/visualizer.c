#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "elmann_rnn.c"

/*
 * visualizer.c
 *
 * Interactive window for the Elman RNN + Genetic Algorithm.
 * Built to be readable and explorable, not just pretty.
 *
 * Controls:
 *   SPACE          pause or resume evolution
 *   R              reset everything
 *   H              toggle hidden state memory panel
 *   [ and ]        fewer or more hidden neurons (resets on change)
 *   + and -        slower or faster evolution speed
 *   click a node   highlight all its connections
 *   hover a node   tooltip showing what it does and its current value
 *   type 0, 1, 2   while paused: feed that number in manually and watch
 *
 * Compile (Mac):
 *   gcc visualizer.c -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lm -o visualizer
 * Run:
 *   ./visualizer
 */

#define POP_SIZE      50
#define GENERATIONS   100
#define MUTATION_RATE 0.05

#define SW 1440
#define SH  860

#define MIN_HIDDEN 2
#define MAX_HIDDEN 12

int h_count = HIDDEN_NEURONS;

typedef struct {
    double gene[TOTAL_WEIGHTS];
    double fitness;
} Chromosome;

Chromosome population[POP_SIZE];
Chromosome new_population[POP_SIZE];

void init_population()
{
    int total = h_count*(INPUT_NEURONS+1) + h_count*h_count + OUTPUT_NEURONS*h_count;
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < total; j++)
            population[i].gene[j] = ((double)rand()/RAND_MAX)*2.0-1.0;
        population[i].fitness = 0.0;
    }
}

void load_weights(double *gene)
{
    int k = 0;
    for (int i = 0; i < h_count; i++)
        for (int j = 0; j < INPUT_NEURONS+1; j++)
            w_input_hidden[i][j] = gene[k++];
    for (int i = 0; i < h_count; i++)
        for (int j = 0; j < h_count; j++)
            w_hidden_hidden[i][j] = gene[k++];
    for (int i = 0; i < OUTPUT_NEURONS; i++)
        for (int j = 0; j < h_count; j++)
            w_hidden_output[i][j] = gene[k++];
}

int select_parent()
{
    int a = rand()%POP_SIZE, b = rand()%POP_SIZE;
    return (population[a].fitness > population[b].fitness) ? a : b;
}

void reproduce()
{
    int total = h_count*(INPUT_NEURONS+1)+h_count*h_count+OUTPUT_NEURONS*h_count;
    for (int i = 0; i < POP_SIZE; i++) {
        int p1 = select_parent(), p2 = select_parent();
        for (int j = 0; j < total; j++) {
            new_population[i].gene[j] = (rand()%2) ?
                population[p1].gene[j] : population[p2].gene[j];
            if (((double)rand()/RAND_MAX) < MUTATION_RATE)
                new_population[i].gene[j] += ((double)rand()/RAND_MAX)*0.2-0.1;
        }
        new_population[i].fitness = 0.0;
    }
    for (int i = 0; i < POP_SIZE; i++) population[i] = new_population[i];
}

void feed_forward_rt()
{
    for (int i = 0; i < h_count; i++) {
        double s = 0.0;
        for (int j = 0; j < INPUT_NEURONS+1; j++) s += w_input_hidden[i][j]*input[j];
        for (int j = 0; j < h_count; j++)         s += w_hidden_hidden[i][j]*context[j];
        hidden[i] = tanh(s);
    }
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        double s = 0.0;
        for (int j = 0; j < h_count; j++) s += w_hidden_output[i][j]*hidden[j];
        outputs[i] = 1.0/(1.0+exp(-s));
    }
    for (int i = 0; i < h_count; i++) context[i] = hidden[i];
}

void reset_ctx() { for (int i = 0; i < h_count; i++) context[i] = 0.0; }



/* Custom sequence the user can edit */
#define MAX_SEQ 16
int custom_seq[MAX_SEQ] = {0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0};
int custom_len = 9;
int editing_seq = 0;  /* 1 = user is typing a new sequence */
char seq_input[MAX_SEQ+1] = "012012012";
int seq_cursor = 9;

void evaluate_population()
{
    int *seq = custom_seq; int len = custom_len;
    for (int i = 0; i < POP_SIZE; i++) {
        load_weights(population[i].gene); reset_ctx();
        double err = 0.0;
        for (int t = 0; t < len-1; t++) {
            for (int j = 0; j < INPUT_NEURONS+1; j++) input[j]=0.0;
            input[0]=1.0; input[seq[t]+1]=1.0;
            feed_forward_rt();
            for (int k = 0; k < OUTPUT_NEURONS; k++) {
                double ex=(k==seq[t+1])?1.0:0.0, d=ex-outputs[k];
                err += d*d;
            }
        }
        population[i].fitness = 1.0/(1.0+err);
    }
}

int find_best()
{
    int b = 0;
    for (int i = 1; i < POP_SIZE; i++)
        if (population[i].fitness > population[b].fitness) b = i;
    return b;
}

/* Colors */
#define C_BG     (Color){13,13,24,255}
#define C_PANEL  (Color){20,20,38,255}
#define C_BORDER (Color){55,55,95,255}
#define C_TITLE  (Color){190,190,230,255}
#define C_GRAY   (Color){90,90,120,255}
#define C_INPUT  (Color){50,130,215,255}
#define C_HIDDEN (Color){210,130,40,255}
#define C_OUTPUT (Color){50,200,110,255}
#define C_CTX    (Color){170,60,215,255}
#define C_BIAS   (Color){90,90,180,255}

#define NX  20
#define NY  50
#define NW 520
#define NH 720
#define NR  15

int COL_I, COL_H, COL_O;

#define CTX_W 155
int CTX_X;
int show_ctx = 1;

#define RX  740
#define RY   50
#define RW  680
#define RH  290

#define PNX 740
#define PNY 360
#define PNW 680
#define PNH 280

#define IFX 740
#define IFY 660
#define IFW 680
#define IFH 160

Vector2 inp_pos[INPUT_NEURONS+1];
Vector2 hid_pos[MAX_HIDDEN];
Vector2 out_pos[OUTPUT_NEURONS];
Vector2 ctx_pos[MAX_HIDDEN];

float inp_act[INPUT_NEURONS+1];
float hid_act[MAX_HIDDEN];
float out_act[OUTPUT_NEURONS];
float ctx_act[MAX_HIDDEN];

float fit_history[GENERATIONS+2];
int   fit_count = 0;

int current_gen = 0;
int paused      = 1;
int done        = 0;
int sel_layer=-1, sel_idx=-1;
int hov_layer=-1, hov_idx=-1;

int demo_pred[MAX_SEQ];
int demo_ready = 0;

int speed_level = 1;
float speed_intervals[] = {0.6f, 0.2f, 0.02f};
const char *speed_labels[] = {"Slow","Medium","Fast"};

int manual_input = -1;

void recalc_layout()
{
    COL_I = NX+55; COL_H = NX+NW/2; COL_O = NX+NW-55;
    CTX_X = NX+NW+8;
    int n = INPUT_NEURONS+1;
    for (int i=0;i<n;i++){
        inp_pos[i].x=COL_I;
        inp_pos[i].y=NY+60+i*(float)(NH-100)/(n-1);
    }
    for (int i=0;i<h_count;i++){
        hid_pos[i].x=COL_H;
        hid_pos[i].y=NY+60+i*(float)(NH-100)/(h_count>1?h_count-1:1);
    }
    for (int i=0;i<OUTPUT_NEURONS;i++){
        out_pos[i].x=COL_O;
        out_pos[i].y=NY+100+i*(float)(NH-200)/(OUTPUT_NEURONS>1?OUTPUT_NEURONS-1:1);
    }
    for (int i=0;i<h_count;i++){
        ctx_pos[i].x=CTX_X+CTX_W/2;
        ctx_pos[i].y=NY+60+i*(float)(NH-100)/(h_count>1?h_count-1:1);
    }
}

Color wcolor(double w, unsigned char a)
{
    float v=fminf(1.0f,(float)fabs(w));
    if(w>=0) return (Color){(unsigned char)(v*230),55,55,a};
    else     return (Color){55,55,(unsigned char)(v*230),a};
}
float wthick(double w){ return fmaxf(0.5f,fminf(4.5f,(float)fabs(w)*3.0f)); }

void draw_panel(int x,int y,int w,int h,const char *t)
{
    DrawRectangle(x,y,w,h,C_PANEL);
    DrawRectangleLines(x,y,w,h,C_BORDER);
    if(t){
        DrawRectangle(x+1,y+1,w-2,24,(Color){28,28,52,255});
        DrawLine(x,y+25,x+w,y+25,C_BORDER);
        DrawText(t,x+8,y+6,13,C_TITLE);
    }
}

void draw_node(Vector2 p,float act,Color col,const char *lbl,int hi,int hov)
{
    if(hi){ DrawCircleV(p,NR+9,Fade(YELLOW,0.18f)); DrawCircleV(p,NR+5,Fade(YELLOW,0.38f)); }
    if(hov&&!hi) DrawCircleV(p,NR+7,Fade(WHITE,0.15f));
    if(act>0.05f) DrawCircleV(p,NR+6,Fade(col,act*0.28f));
    DrawCircleV(p,NR,col);
    DrawCircleLines((int)p.x,(int)p.y,NR,hov?WHITE:(Color){180,180,180,160});
    if(act>0.01f) DrawCircleSector(p,NR-3,-90,-90+act*360,32,Fade(WHITE,0.20f));
    if(lbl){ int tw=MeasureText(lbl,9); DrawText(lbl,(int)(p.x-tw/2),(int)(p.y-5),9,WHITE); }
}

int near_node(Vector2 m,Vector2 p){ return CheckCollisionPointCircle(m,p,NR+8); }

void update_activations()
{
    for(int i=0;i<INPUT_NEURONS+1;i++) inp_act[i]=(float)fabs(input[i]);
    for(int i=0;i<h_count;i++) hid_act[i]=(float)((hidden[i]+1.0)/2.0);
    for(int i=0;i<OUTPUT_NEURONS;i++) out_act[i]=(float)outputs[i];
    for(int i=0;i<h_count;i++) ctx_act[i]=(float)((context[i]+1.0)/2.0);
}

void run_demo(int best)
{
    load_weights(population[best].gene); reset_ctx();
    int steps = custom_len-1;
    for(int t=0;t<steps;t++){
        for(int j=0;j<INPUT_NEURONS+1;j++) input[j]=0.0;
        input[0]=1.0; input[custom_seq[t]+1]=1.0;
        feed_forward_rt();
        int pred=0;
        for(int k=1;k<OUTPUT_NEURONS;k++) if(outputs[k]>outputs[pred]) pred=k;
        demo_pred[t]=pred;
    }
    update_activations();
    demo_ready=1;
}

void draw_network()
{
    draw_panel(NX,NY,NW,NH,"NETWORK  (hover a node to learn about it, click to inspect connections)");
    DrawText("INPUT", COL_I-18,NY+30,11,C_GRAY);
    DrawText("HIDDEN",COL_H-22,NY+30,11,C_GRAY);
    DrawText("OUTPUT",COL_O-22,NY+30,11,C_GRAY);
    int n=INPUT_NEURONS+1;

    for(int i=0;i<h_count;i++)
        for(int j=0;j<n;j++){
            double w=w_input_hidden[i][j];
            int hi=(sel_layer==0&&sel_idx==j)||(sel_layer==1&&sel_idx==i);
            DrawLineEx(inp_pos[j],hid_pos[i],hi?wthick(w)+1:0.6f,wcolor(w,hi?200:25));
        }
    for(int i=0;i<OUTPUT_NEURONS;i++)
        for(int j=0;j<h_count;j++){
            double w=w_hidden_output[i][j];
            int hi=(sel_layer==1&&sel_idx==j)||(sel_layer==2&&sel_idx==i);
            DrawLineEx(hid_pos[j],out_pos[i],hi?wthick(w)+1:0.6f,wcolor(w,hi?200:25));
        }
    for(int i=0;i<h_count;i++)
        for(int j=0;j<h_count;j++){
            if(i==j) continue;
            double w=w_hidden_hidden[i][j];
            int hi=(sel_layer==1&&(sel_idx==i||sel_idx==j));
            DrawLineBezier(hid_pos[j],hid_pos[i],hi?1.6f:0.3f,wcolor(w,hi?140:12));
        }

    const char *il[]={"bias","in:0","in:1","in:2","in:3","in:4","in:5"};
    for(int i=0;i<n;i++){
        int hi=(sel_layer==0&&sel_idx==i), hov=(hov_layer==0&&hov_idx==i);
        draw_node(inp_pos[i],inp_act[i],(i==0)?C_BIAS:C_INPUT,il[i],hi,hov);
    }

    char lbl[8];
    for(int i=0;i<h_count;i++){
        int hi=(sel_layer==1&&sel_idx==i), hov=(hov_layer==1&&hov_idx==i);
        snprintf(lbl,sizeof(lbl),"h%d",i);
        draw_node(hid_pos[i],hid_act[i],C_HIDDEN,lbl,hi,hov);
        if(show_ctx){
            Vector2 tip={hid_pos[i].x+NR+1,hid_pos[i].y};
            Vector2 end={(float)CTX_X-2,ctx_pos[i].y};
            DrawLineEx(tip,end,0.5f,Fade(C_CTX,0.18f));
        }
    }

    int winner=0;
    for(int i=1;i<OUTPUT_NEURONS;i++) if(out_act[i]>out_act[winner]) winner=i;
    const char *ol[]={"out:0","out:1","out:2","out:3","out:4","out:5"};
    for(int i=0;i<OUTPUT_NEURONS;i++){
        int hi=(sel_layer==2&&sel_idx==i), hov=(hov_layer==2&&hov_idx==i);
        Color c=(demo_ready&&i==winner)?(Color){80,255,140,255}:C_OUTPUT;
        draw_node(out_pos[i],out_act[i],c,ol[i],hi,hov);
        char vl[8]; snprintf(vl,sizeof(vl),"%.2f",outputs[i]);
        DrawText(vl,(int)out_pos[i].x+NR+4,(int)out_pos[i].y-5,9,
            (i==winner&&demo_ready)?LIME:C_GRAY);
    }

    /* Legend */
    int lx=NX+8,ly=NY+NH+2;
    DrawRectangle(lx,     ly,10,7,(Color){220,55,55,200}); DrawText("positive weight",lx+13,ly-1,9,C_GRAY);
    DrawRectangle(lx+140, ly,10,7,(Color){55,55,220,200}); DrawText("negative weight",lx+153,ly-1,9,C_GRAY);
    DrawText("brightest output = prediction",lx+290,ly-1,9,C_GRAY);
}

void draw_context()
{
    if(!show_ctx) return;
    draw_panel(CTX_X,NY,CTX_W,NH,"HIDDEN STATE");
    DrawText("memory from",  CTX_X+6,NY+28,9,C_GRAY);
    DrawText("the last step",CTX_X+6,NY+39,9,C_GRAY);
    char lbl[10],val[10];
    for(int i=0;i<h_count;i++){
        int hi=(sel_layer==1&&sel_idx==i), hov=(hov_layer==1&&hov_idx==i);
        snprintf(lbl,sizeof(lbl),"ctx%d",i);
        draw_node(ctx_pos[i],ctx_act[i],C_CTX,lbl,hi,hov);
        Vector2 src={ctx_pos[i].x-NR-1,ctx_pos[i].y};
        Vector2 dst={(float)COL_H+NR+1,hid_pos[i].y};
        DrawLineEx(src,dst,hi?1.8f:0.5f,Fade(C_CTX,hi?0.75f:0.18f));
        snprintf(val,sizeof(val),"%.2f",context[i]);
        DrawText(val,(int)ctx_pos[i].x+NR+4,(int)ctx_pos[i].y-5,9,Fade(WHITE,0.45f));
    }
    DrawText("These values come",CTX_X+4,NY+NH-52,9,C_GRAY);
    DrawText("from the previous",CTX_X+4,NY+NH-40,9,C_GRAY);
    DrawText("step and feed back",CTX_X+4,NY+NH-28,9,C_GRAY);
    DrawText("in as memory.",    CTX_X+4,NY+NH-16,9,C_GRAY);
    DrawText("Press H to hide.", CTX_X+4,NY+NH-4, 9,C_GRAY);
}

void draw_fitness_graph()
{
    draw_panel(RX,RY,RW,RH,"FITNESS OVER GENERATIONS");
    DrawText("1.0 = perfect predictions,  0.0 = completely wrong",RX+8,RY+28,11,C_GRAY);
    int mx=RX+40,my=RY+48,mw=RW-52,mh=RH-68;
    DrawLine(mx,my,mx,my+mh,C_BORDER);
    DrawLine(mx,my+mh,mx+mw,my+mh,C_BORDER);
    DrawText("1.0",RX+6,my-6,     10,C_GRAY);
    DrawText("0.5",RX+6,my+mh/2-6,10,C_GRAY);
    DrawText("0.0",RX+6,my+mh-6,  10,C_GRAY);
    DrawLine(mx,my+mh/2,mx+mw,my+mh/2,(Color){38,38,62,255});
    DrawText("gen 0",mx-10,my+mh+4,9,C_GRAY);
    char gl[12]; snprintf(gl,sizeof(gl),"gen %d",GENERATIONS);
    DrawText(gl,mx+mw-30,my+mh+4,9,C_GRAY);

    if(fit_count<2){
        DrawText("Press SPACE to start",mx+mw/2-60,my+mh/2-8,13,C_GRAY);
    } else {
        float xs=(float)mw/GENERATIONS, ys=(float)mh;
        for(int i=1;i<fit_count;i++){
            float x1=mx+(i-1)*xs, y1=my+mh-fit_history[i-1]*ys;
            float x2=mx+i    *xs, y2=my+mh-fit_history[i]    *ys;
            DrawLineEx((Vector2){x1,y1},(Vector2){x2,y2},2.2f,GREEN);
        }
        char fl[32]; snprintf(fl,sizeof(fl),"Best: %.4f",fit_history[fit_count-1]);
        DrawText(fl,RX+RW-140,RY+8,13,LIME);
    }

    /* Progress bar */
    float prog=(float)current_gen/GENERATIONS;
    DrawRectangle(mx,my+mh+18,mw,8,(Color){30,30,50,255});
    DrawRectangle(mx,my+mh+18,(int)(mw*prog),8,(Color){60,180,100,200});
}

void draw_predictions()
{
    draw_panel(PNX,PNY,PNW,PNH,"SEQUENCE PREDICTIONS  (best network this generation)");

    /* Show the current sequence and allow editing */
    int steps = custom_len - 1;
    char seq_display[64] = "Sequence: ";
    for(int i=0;i<custom_len;i++){
        char c[3]; snprintf(c,sizeof(c),"%d ",custom_seq[i]);
        strncat(seq_display,c,sizeof(seq_display)-strlen(seq_display)-1);
    }
    DrawText(seq_display,PNX+8,PNY+28,11,C_GRAY);

    if(editing_seq){
        DrawRectangle(PNX+8,PNY+44,PNW-16,22,(Color){30,30,55,255});
        DrawRectangleLines(PNX+8,PNY+44,PNW-16,22,YELLOW);
        char prompt[80]; snprintf(prompt,sizeof(prompt),"Type sequence (0-2 only, max %d digits, ENTER to confirm): %s_",MAX_SEQ,seq_input);
        DrawText(prompt,PNX+12,PNY+49,11,YELLOW);
        return;
    }

    DrawText("Press E to edit sequence  (digits 0-2 only, e.g. 012012)",PNX+8,PNY+44,10,C_GRAY);

    if(!demo_ready){
        DrawText("Press SPACE to start. Predictions appear after generation 1.",PNX+12,PNY+72,12,C_GRAY);
        return;
    }

    DrawText("Step",     PNX+12, PNY+62,11,C_GRAY);
    DrawText("Input",    PNX+70, PNY+62,11,C_GRAY);
    DrawText("Predicted",PNX+150,PNY+62,11,C_GRAY);
    DrawText("Expected", PNX+240,PNY+62,11,C_GRAY);
    DrawText("Result",   PNX+330,PNY+62,11,C_GRAY);
    DrawLine(PNX+8,PNY+76,PNX+PNW-8,PNY+76,C_BORDER);

    int ok_count=0;
    int max_show = steps < 8 ? steps : 8;
    for(int t=0;t<max_show;t++){
        int ok=(demo_pred[t]==custom_seq[t+1]); if(ok) ok_count++;
        int ry=PNY+82+t*24;
        Color rc=ok?(Color){70,210,110,255}:(Color){210,70,70,255};
        DrawRectangle(PNX+8,ry-1,PNW-16,20,ok?(Color){25,55,30,80}:(Color){55,25,25,80});
        char tmp[8];
        snprintf(tmp,sizeof(tmp),"%d",t+1);               DrawText(tmp,PNX+20, ry+3,11,C_GRAY);
        snprintf(tmp,sizeof(tmp),"%d",custom_seq[t]);      DrawText(tmp,PNX+80, ry+3,12,WHITE);
        snprintf(tmp,sizeof(tmp),"%d",demo_pred[t]);        DrawText(tmp,PNX+162,ry+3,12,rc);
        snprintf(tmp,sizeof(tmp),"%d",custom_seq[t+1]);    DrawText(tmp,PNX+252,ry+3,12,WHITE);
        DrawText(ok?"correct":"wrong",PNX+330,ry+3,11,rc);
    }
    char sc[32]; snprintf(sc,sizeof(sc),"%d / %d correct",ok_count,max_show);
    DrawText(sc,PNX+PNW-130,PNY+8,13,
        ok_count==max_show?LIME:(ok_count>=max_show/2?YELLOW:(Color){200,80,80,255}));
}

void draw_info()
{
    draw_panel(IFX,IFY,IFW,IFH,"INFO");

    char hc[48]; snprintf(hc,sizeof(hc),"Hidden neurons: %d  (use [ ] to change)",h_count);
    DrawText(hc,IFX+8,IFY+32,11,C_TITLE);

    char sp[48]; snprintf(sp,sizeof(sp),"Speed: %s  (use + - to change)",speed_labels[speed_level]);
    DrawText(sp,IFX+280,IFY+32,11,C_TITLE);

    DrawText("SPACE start/pause    R reset    H memory    E edit sequence    0/1/2 manual test    [ ] neurons    + - speed",
        IFX+8,IFY+52,10,C_GRAY);

    DrawLine(IFX+8,IFY+68,IFX+IFW-8,IFY+68,C_BORDER);

    if(hov_layer==-1){
        DrawText("Hover any node to see what it does and its current value.",IFX+8,IFY+78,12,C_GRAY);
        if(paused&&!done&&demo_ready)
            DrawText("You are paused. Type 0, 1, or 2 to manually feed a number and watch the network react.",
                IFX+8,IFY+98,11,C_GRAY);
    } else {
        const char *desc="";
        char val_line[100]="";
        if(hov_layer==0){
            if(hov_idx==0){
                desc="BIAS — always 1.0. Gives every hidden neuron a constant baseline to shift its output.";
                snprintf(val_line,sizeof(val_line),"Value: 1.0 (constant)");
            } else {
                desc="INPUT NEURON — one-hot encoded. It fires (1.0) when its number is the current step, otherwise 0.";
                snprintf(val_line,sizeof(val_line),"Current value: %.3f",input[hov_idx]);
            }
        } else if(hov_layer==1){
            desc="HIDDEN NEURON — mixes current input with memory from last step. Uses tanh, so output is -1 to +1.";
            snprintf(val_line,sizeof(val_line),"Activation now: %.3f     Memory (context) from last step: %.3f",
                hidden[hov_idx],context[hov_idx]);
        } else {
            desc="OUTPUT NEURON — probability this is the next number. Uses sigmoid so output is 0 to 1. Highest one wins.";
            snprintf(val_line,sizeof(val_line),"Probability: %.3f",outputs[hov_idx]);
        }
        DrawText(desc,    IFX+8,IFY+78,11,C_TITLE);
        DrawText(val_line,IFX+8,IFY+98,11,LIME);
    }

    if(manual_input>=0){
        char mi[80];
        snprintf(mi,sizeof(mi),"You fed %d into the network manually. Watch which output lights up brightest.",manual_input);
        DrawText(mi,IFX+8,IFY+122,11,YELLOW);
    }
}

void draw_status()
{
    DrawRectangle(0,SH-28,SW,28,(Color){16,16,30,255});
    DrawLine(0,SH-28,SW,SH-28,C_BORDER);
    char s[180];
    if(done)
        snprintf(s,sizeof(s),"Done. %d generations. Best fitness: %.4f   Press R to restart.",
            GENERATIONS,fit_history[fit_count-1]);
    else if(paused&&current_gen==0)
        snprintf(s,sizeof(s),"Ready. Press SPACE to start evolution. Hover nodes to learn about them first.");
    else if(paused)
        snprintf(s,sizeof(s),"Paused gen %d / %d.   SPACE go.   0/1/2 test manually.   E edit sequence.   [ ] neurons.   + - speed.",
            current_gen,GENERATIONS);
    else
        snprintf(s,sizeof(s),"Evolving — gen %d / %d    SPACE pause    R reset    H memory    [ ] neurons    + - speed    E edit sequence",
            current_gen,GENERATIONS);
    DrawText(s,10,SH-20,12,(Color){170,170,195,255});
}

void handle_input()
{
    if(IsKeyPressed(KEY_SPACE)){ paused=!paused; manual_input=-1; }

    if(IsKeyPressed(KEY_R)){
        current_gen=0; fit_count=0; done=0; demo_ready=0;
        paused=1; manual_input=-1; sel_layer=-1; sel_idx=-1;
        reset_ctx(); init_population(); recalc_layout();
    }

    if(IsKeyPressed(KEY_H)){ show_ctx=!show_ctx; recalc_layout(); }

    if(IsKeyPressed(KEY_EQUAL))  { if(speed_level<2) speed_level++; }
    if(IsKeyPressed(KEY_MINUS))  { if(speed_level>0) speed_level--; }

    if(IsKeyPressed(KEY_RIGHT_BRACKET)&&h_count<MAX_HIDDEN){
        h_count++; current_gen=0; fit_count=0; done=0; demo_ready=0;
        paused=1; reset_ctx(); init_population(); recalc_layout();
    }
    if(IsKeyPressed(KEY_LEFT_BRACKET)&&h_count>MIN_HIDDEN){
        h_count--; current_gen=0; fit_count=0; done=0; demo_ready=0;
        paused=1; reset_ctx(); init_population(); recalc_layout();
    }

    /* Sequence editing mode: press E to start, type digits 0-2, ENTER to confirm, ESC to cancel */
    if(IsKeyPressed(KEY_E)&&paused&&!editing_seq){
        editing_seq=1;
        seq_cursor=0; seq_input[0]=0;
    }
    if(editing_seq){
        /* Accept digits 0, 1, 2 */
        int digit=-1;
        if(IsKeyPressed(KEY_ZERO)) digit=0;
        if(IsKeyPressed(KEY_ONE))  digit=1;
        if(IsKeyPressed(KEY_TWO))  digit=2;
        if(digit>=0 && seq_cursor<MAX_SEQ){
            seq_input[seq_cursor++]='0'+digit;
            seq_input[seq_cursor]=0;
        }
        /* Backspace */
        if(IsKeyPressed(KEY_BACKSPACE)&&seq_cursor>0){
            seq_cursor--; seq_input[seq_cursor]=0;
        }
        /* Confirm */
        if(IsKeyPressed(KEY_ENTER)&&seq_cursor>=2){
            custom_len=seq_cursor;
            for(int i=0;i<custom_len;i++) custom_seq[i]=seq_input[i]-'0';
            editing_seq=0;
            /* Reset so network trains on new sequence */
            current_gen=0; fit_count=0; done=0; demo_ready=0;
            reset_ctx(); init_population();
        }
        /* Cancel */
        if(IsKeyPressed(KEY_ESCAPE)) editing_seq=0;
        return; /* eat all other keys while editing */
    }

    if(paused&&!done&&demo_ready){
        int feed=-1;
        if(IsKeyPressed(KEY_ZERO)) feed=0;
        if(IsKeyPressed(KEY_ONE))  feed=1;
        if(IsKeyPressed(KEY_TWO))  feed=2;
        if(feed>=0){
            manual_input=feed;
            double saved[MAX_HIDDEN];
            for(int i=0;i<h_count;i++) saved[i]=context[i];
            for(int j=0;j<INPUT_NEURONS+1;j++) input[j]=0.0;
            input[0]=1.0; input[feed+1]=1.0;
            feed_forward_rt();
            update_activations();
            for(int i=0;i<h_count;i++) context[i]=saved[i];
        }
    }

    if(IsMouseButtonPressed(MOUSE_LEFT_BUTTON)){
        Vector2 m=GetMousePosition(); int found=0;
        for(int i=0;i<INPUT_NEURONS+1&&!found;i++)
            if(near_node(m,inp_pos[i])){ sel_layer=0; sel_idx=i; found=1; }
        for(int i=0;i<h_count&&!found;i++)
            if(near_node(m,hid_pos[i])){ sel_layer=1; sel_idx=i; found=1; }
        for(int i=0;i<OUTPUT_NEURONS&&!found;i++)
            if(near_node(m,out_pos[i])){ sel_layer=2; sel_idx=i; found=1; }
        if(!found){ sel_layer=-1; sel_idx=-1; }
    }

    {
        Vector2 m=GetMousePosition(); hov_layer=-1; hov_idx=-1;
        for(int i=0;i<INPUT_NEURONS+1;i++)
            if(near_node(m,inp_pos[i])){ hov_layer=0; hov_idx=i; break; }
        if(hov_layer==-1)
            for(int i=0;i<h_count;i++)
                if(near_node(m,hid_pos[i])){ hov_layer=1; hov_idx=i; break; }
        if(hov_layer==-1)
            for(int i=0;i<OUTPUT_NEURONS;i++)
                if(near_node(m,out_pos[i])){ hov_layer=2; hov_idx=i; break; }
    }
}

int main()
{
    srand((unsigned int)time(NULL));
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_WINDOW_HIGHDPI);
    InitWindow(SW,SH,"Elman RNN + Genetic Algorithm — Interactive Visualizer");
    SetTargetFPS(60);
    recalc_layout(); reset_ctx(); init_population();
    float gen_timer=0.0f;

    while(!WindowShouldClose()){
        float dt=GetFrameTime();
        handle_input();
        if(!paused&&!done){
            gen_timer+=dt;
            if(gen_timer>=speed_intervals[speed_level]){
                gen_timer=0.0f;
                evaluate_population();
                int best=find_best();
                fit_history[fit_count++]=(float)population[best].fitness;
                run_demo(best);
                reproduce();
                current_gen++;
                if(current_gen>=GENERATIONS) done=1;
            }
        }
        BeginDrawing();
        ClearBackground(C_BG);
        draw_network();
        draw_context();
        draw_fitness_graph();
        draw_predictions();
        draw_info();
        draw_status();
        EndDrawing();
    }
    CloseWindow();
    return 0;
}