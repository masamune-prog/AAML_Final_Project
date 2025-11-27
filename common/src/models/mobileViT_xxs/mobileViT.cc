#include "models/mobileViT_xxs/mobileViT.h"
#include "models/mobileViT_xxs/mobileViT_xxs.h"
#include <stdio.h>
#include "menu.h"
#include "tflite.h"

// Initialize everything once
// deallocate tensors when done
static void mobileViT_xxs_init(void) {
  tflite_load_model(mobileViT_xxs, mobileViT_xxs_len);
}

static int8_t* non_stream_classify() {
    printf("Running mobileViT_xxs\n");
    tflite_classify();
    // Process the inference results.
    int8_t* output = tflite_get_output();
    return output;
}

static void do_classify() {
    tflite_set_input_zeros();
    int8_t* result = non_stream_classify();
    for(size_t i=0; i<10; i++)
        printf("%d : %d,\n", i, result[i]);
}

static void do_classify_random() {
    tflite_randomize_input(8888);
    int8_t* result = non_stream_classify();
    for(size_t i=0; i<10; i++)
        printf("%d : %d,\n", i, result[i]);
}

static struct Menu MENU = {
    "Tests for mobileViT_xxs",
    "mobileViT_xxs",
    {
        MENU_ITEM('1', "Run with zeros input", do_classify),
        MENU_ITEM('2', "Run with random input", do_classify_random),
        MENU_END,
    },
};

// For integration into menu system
void mobileViT_xxs_menu() {
  mobileViT_xxs_init();
  menu_run(&MENU);
}