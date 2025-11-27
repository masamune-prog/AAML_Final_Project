#include "models/ds_cnn_stream_fe/ds_cnn.h"
#include <stdio.h>
#include "menu.h"
#include "models/ds_cnn_stream_fe/ds_cnn_stream_fe.h"
#include "tflite.h"
#include "models/label/label/label0_board.h"
#include "models/label/label/label1_board.h"
#include "models/label/label/label6_board.h"
#include "models/label/label/label8_board.h"
#include "models/label/label/label11_board.h"



// The cnn model classifies 0-11 based on the greatest of 12 scores.
typedef struct {
  uint32_t score_0;  // Stored as uint32_t because we can't print floats.
  uint32_t score_1;
  uint32_t score_2; 
  uint32_t score_3;
  uint32_t score_4;
  uint32_t score_5;
  uint32_t score_6;
  uint32_t score_7;
  uint32_t score_8;
  uint32_t score_9;
  uint32_t score_10;
  uint32_t score_11;
} cnnResult;


// Initialize everything once
// deallocate tensors when done
static void ds_cnn_stream_fe_init(void) {
  tflite_load_model(ds_cnn_stream_fe, ds_cnn_stream_fe_len);
}

// TODO: Implement your design here
// Run classification, after input has been loaded
cnnResult ds_cnn_stream_fe_classify() {
  printf("Running ds_cnn_stream_fe\n");
  tflite_classify();

  // Process the inference results.
  float* output = tflite_get_output_float();

  // Get the raw bits of the floats.
  return (cnnResult){
      *(uint32_t*)&output[0],
      *(uint32_t*)&output[1],
      *(uint32_t*)&output[2],
      *(uint32_t*)&output[3],
      *(uint32_t*)&output[4],
      *(uint32_t*)&output[5],
      *(uint32_t*)&output[6],
      *(uint32_t*)&output[7],
      *(uint32_t*)&output[8],
      *(uint32_t*)&output[9],
      *(uint32_t*)&output[10],
      *(uint32_t*)&output[11],
  };
}

static void print_ds_cnn_stream_fe_result(const char* prefix, cnnResult res) {
  printf("%s-- 0: 0x%lx\n", prefix, res.score_0);
  printf("%s-- 1: 0x%lx\n", prefix, res.score_1);
  printf("%s-- 2: 0x%lx\n", prefix, res.score_2);
  printf("%s-- 3: 0x%lx\n", prefix, res.score_3);
  printf("%s-- 4: 0x%lx\n", prefix, res.score_4);
  printf("%s-- 5: 0x%lx\n", prefix, res.score_5);
  printf("%s-- 6: 0x%lx\n", prefix, res.score_6);
  printf("%s-- 7: 0x%lx\n", prefix, res.score_7);
  printf("%s-- 8: 0x%lx\n", prefix, res.score_8);
  printf("%s-- 9: 0x%lx\n", prefix, res.score_9);
  printf("%s-- 10: 0x%lx\n", prefix, res.score_10);
  printf("%s-- 11: 0x%lx\n", prefix, res.score_11);
}
static void do_classify_0() {
  puts("Classify 0");
  tflite_set_input_float(label0_data);
  print_ds_cnn_stream_fe_result("  results", ds_cnn_stream_fe_classify());
}
static void do_classify_1() {
  puts("Classify 1");
  tflite_set_input_float(label1_data);
  print_ds_cnn_stream_fe_result("  results", ds_cnn_stream_fe_classify());
}
static void do_classify_6() {
  puts("Classify 6");
  tflite_set_input_float(label6_data);
  print_ds_cnn_stream_fe_result("  results", ds_cnn_stream_fe_classify());
}
static void do_classify_8() {
  puts("Classify 8");
  tflite_set_input_float(label8_data);
  print_ds_cnn_stream_fe_result("  results", ds_cnn_stream_fe_classify());
}
static void do_classify_11() {
  puts("Classify 11");
  tflite_set_input_float(label11_data);
  print_ds_cnn_stream_fe_result("  results", ds_cnn_stream_fe_classify());
}

//we test the model with the following inputs 0,1,6,8,11
static struct Menu MENU = {
    "Tests for ds_cnn_stream_fe",
    "ds_cnn_stream_fe",
    {
        MENU_ITEM('1', "Run with 0 input", do_classify_0),
        MENU_ITEM('2', "Run with 1 input", do_classify_1),
        MENU_ITEM('3', "Run with 6 input", do_classify_6),
        MENU_ITEM('4', "Run with 8 input", do_classify_8),
        MENU_ITEM('5', "Run with 11 input", do_classify_11),
        MENU_END,
    },
};

// For integration into menu system
void ds_cnn_stream_fe_menu() {
  ds_cnn_stream_fe_init();
  menu_run(&MENU);
}
