Project title. Institutional Water Bill Prediction System  
Project Title
Design and Implementation of a Water Bill Prediction System Using Machine Learning
Problem Statement
Many institutions and households experience challenges in budgeting for water expenses because water bills are only known after consumption has occurred. Existing water billing systems mainly record water usage and generate bills but do not predict future water costs. This makes it difficult for users to plan their finances, detect unusual consumption trends, and make informed decisions regarding water usage.
Proposed Solution
The Water Bill Prediction System uses machine learning techniques to analyse historical water consumption data and predict future water bills. The system collects water flow data through a Hall-effect water flow sensor connected to a Raspberry Pi, stores the data in a database, and applies machine learning algorithms to estimate future water bills.
The system enables users to:
Monitor water consumption.
View historical usage records.
Predict future monthly water bills.
Support budgeting and financial planning.
Detect unusual water consumption trends.
Technologies Used
Hardware
Raspberry Pi
Hall-effect Water Flow Sensor
Latching Valve
Lithium-ion Battery
Printed Circuit Board (PCB)
Software
Python
Flask
MySQL
HTML
CSS
JavaScript
Visual Studio Code
Git
Machine Learning Libraries
Scikit-learn
Pandas
NumPy
Matplotlib
System Requirements
Hardware Requirements
Raspberry Pi 4 Model B
Hall-effect Water Flow Sensor
Internet Connection
Power Supply
Personal Computer
Software Requirements
Python 3.10 or later
MySQL Server
Git
Visual Studio Code
Web Browser (Google Chrome or Microsoft Edge)Setup Instructions
Download or clone the project from the GitHub repository.
Install the required software, including Python, MySQL, and Visual Studio Code.
Install the required Python packages using the requirements.txt file.
Create the project database in MySQL and import the provided database file.
Update the database connection settings in the project configuration.
Connect the Raspberry Pi to the Hall-effect water flow sensor and ensure all hardware connections are correct.
Run the application using Python.
Open the application in a web browser and log in to access the dashboard.
Start monitoring water consumption and generate water bill predictions./**
 * ============================================================
 *  WATER FLOW NODE 1  –  ATmega328P  (v3 – NRF IRQ driven)
 * ============================================================
 *
 *  Pin Assignment
 *  ──────────────────────────────────────────────────────────
 *  Pin  2  (INT0)   <- YF-S201 flow pulse         [interrupt]
 *  Pin  3  (PWM)    -> RGB LED  Red   + 220Ohm
 *  Pin  4           <- NRF24L01 IRQ               [PCINT20]
 *  Pin  5  (PWM)    -> RGB LED  Green + 220Ohm
 *  Pin  6  (PWM)    -> RGB LED  Blue  + 220Ohm
 *  Pin  8           -> SD card  CS
 *  Pin  9           -> NRF24L01 CE
 *  Pin 10           -> NRF24L01 CSN
 *  Pin 11  (MOSI)   -> SPI bus
 *  Pin 12  (MISO)   <- SPI bus
 *  Pin 13  (SCK)    -> SPI bus
 *  Pin A4  (SDA)    <-> DS3231 RTC
 *  Pin A5  (SCL)    <-> DS3231 RTC
 *
 *  NRF24L01 IRQ Strategy
 *  ──────────────────────────────────────────────────────────
 *  IRQ pin -> ATmega pin 4 (PCINT20, PORTD bit 4).
 *  radio.maskIRQ(true, true, false) -> only RX_DR fires IRQ.
 *  PCINT2_vect ISR sets volatile nrfIrqFired flag.
 *  Main loop reads flag, calls whatHappened(), handles packet.
 *  Zero SPI calls inside any ISR.
 *
 *  RGB LED States
 *  ──────────────────────────────────────────────────────────
 *  IDLE         Cyan    Slow 3s breath     Listening, no flow
 *  FLOW_ACTIVE  Green   Rate-scaled breath Flow detected
 *  REQUEST_RX   Yellow  2x 80ms flash      IRQ fired
 *  TRANSMITTING Blue    50ms strobe        RF TX active
 *  TX_SUCCESS   White   300ms solid        ACK confirmed
 *  TX_FAIL      Red     3x 100ms flash     No ACK
 *  SD_WRITE     Magenta 80ms blip          Writing SD
 *  HW_ERROR     Red     1Hz blink          Hardware fault
 * ============================================================
 */

// ── Includes first ──────────────────────────────────────────
#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <RTClib.h>
#include <RF24.h>
#include <avr/interrupt.h>

// ── ALL TYPE DEFINITIONS BEFORE ANY FUNCTIONS ───────────────
// This block must come before any function definitions so the
// Arduino IDE prototype generator sees them first and does not
// produce invalid forward declarations referencing unknown types.

// RGB colour struct
struct RGB {
  uint8_t r, g, b;
};

// LED state enum
enum LedState : uint8_t {
  LED_IDLE,
  LED_FLOW_ACTIVE,
  LED_REQUEST_RX,
  LED_TRANSMITTING,
  LED_TX_SUCCESS,
  LED_TX_FAIL,
  LED_SD_WRITE,
  LED_HW_ERROR
};

// NRF24 data packet (must match master and node 2)
struct FlowPacket {
  uint8_t nodeId;
  uint32_t timestamp;
  float flowRateLPM;
  float totalLitres;
  uint8_t status;  // bit0=SD_ERR  bit1=RTC_ERR
};

// ──────────────────────────────────────────────
//  Pin definitions
// ──────────────────────────────────────────────
#define FLOW_SENSOR_PIN 2  // INT0  – YF-S201
#define RGB_R_PIN 3        // PWM   – LED Red
#define NRF_IRQ_PIN 4      // PCINT20 – NRF24L01 IRQ (active LOW)
#define RGB_G_PIN 5        // PWM   – LED Green
#define RGB_B_PIN 6        // PWM   – LED Blue
#define SD_CS_PIN 8
#define NRF_CE_PIN 9
#define NRF_CSN_PIN 10

// ──────────────────────────────────────────────
//  RGB LED config
// ──────────────────────────────────────────────
#define RGB_COMMON_ANODE false  // set true for common-anode LED

// ──────────────────────────────────────────────
//  Node identity
// ──────────────────────────────────────────────
#define NODE_ID 1
const uint64_t RX_PIPE = 0xA1A1A1A1A1LL;
const uint64_t TX_PIPE = 0xB0B0B0B0B0LL;

// ──────────────────────────────────────────────
//  Timing
// ──────────────────────────────────────────────
#define SAMPLE_INTERVAL_MS 5000UL
#define CALIBRATION_FACTOR 7.5f
#define FLOW_IDLE_TIMEOUT_MS 8000UL

// ══════════════════════════════════════════════
//  NRF24L01 IRQ  –  PCINT2 on pin 4 (PD4 = PCINT20)
// ══════════════════════════════════════════════
volatile bool nrfIrqFired = false;

ISR(PCINT2_vect) {
  // Active LOW – only act on falling edge
  if (!(PIND & (1 << PD4))) {
    nrfIrqFired = true;
  }
}

void enableNrfIrq() {
  pinMode(NRF_IRQ_PIN, INPUT);  // NRF has internal pull-up on IRQ
  PCICR |= (1 << PCIE2);        // enable PCINT[23:16] group
  PCMSK2 |= (1 << PCINT20);     // unmask pin 4
}

// ══════════════════════════════════════════════
//  RGB colour constants
// ══════════════════════════════════════════════
const RGB COL_OFF = { 0, 0, 0 };
const RGB COL_IDLE = { 0, 80, 80 };    // dim cyan
const RGB COL_FLOW = { 0, 255, 0 };    // green
const RGB COL_REQ = { 255, 200, 0 };   // yellow
const RGB COL_TX = { 0, 0, 255 };      // blue
const RGB COL_OK = { 255, 255, 255 };  // white
const RGB COL_FAIL = { 255, 0, 0 };    // red
const RGB COL_SD = { 180, 0, 180 };    // magenta

// ══════════════════════════════════════════════
//  RGB LED driver
// ══════════════════════════════════════════════
void rgbWrite(uint8_t r, uint8_t g, uint8_t b) {
  if (RGB_COMMON_ANODE) {
    r = 255 - r;
    g = 255 - g;
    b = 255 - b;
  }
  analogWrite(RGB_R_PIN, r);
  analogWrite(RGB_G_PIN, g);
  analogWrite(RGB_B_PIN, b);
}

// Overload taking a colour struct – defined AFTER struct RGB
void rgbWrite(RGB c) {
  rgbWrite(c.r, c.g, c.b);
}

void rgbOff() {
  rgbWrite(0, 0, 0);
}

uint8_t breathe(uint32_t periodMs, uint8_t peak) {
  float t = (float)(millis() % periodMs) / (float)periodMs;
  return (uint8_t)((1.0f - cosf(t * 6.28318f)) * 0.5f * (float)peak);
}

// ══════════════════════════════════════════════
//  LED state machine globals
// ══════════════════════════════════════════════
LedState ledState = LED_IDLE;
LedState ledBaseState = LED_IDLE;
uint32_t ledPhaseMs = 0;
uint8_t ledPhase = 0;

// ── State machine helpers ────────────────────
void enterLedState(LedState s) {
  ledState = s;
  ledPhase = 0;
  ledPhaseMs = millis();
}

void ledSetBase(LedState s) {
  ledBaseState = s;
  enterLedState(s);
}

void ledTransient(LedState s) {
  enterLedState(s);
}

// ── Public convenience wrappers ──────────────
void ledIdle() {
  ledSetBase(LED_IDLE);
}
void ledFlowActive() {
  ledSetBase(LED_FLOW_ACTIVE);
}
void ledRequestRx() {
  ledTransient(LED_REQUEST_RX);
}
void ledTransmitting() {
  ledTransient(LED_TRANSMITTING);
}
void ledTxOk() {
  ledTransient(LED_TX_SUCCESS);
}
void ledTxFail() {
  ledTransient(LED_TX_FAIL);
}
void ledSdWrite() {
  ledTransient(LED_SD_WRITE);
}
void ledHwError() {
  ledSetBase(LED_HW_ERROR);
}

// ── Tick – must be called every loop() iteration ─────────────
void ledTick() {
  uint32_t elapsed = millis() - ledPhaseMs;

  switch (ledState) {

    case LED_IDLE:
      {
        uint8_t v = breathe(3000, 80);
        rgbWrite(0, v, v);
        break;
      }

    case LED_FLOW_ACTIVE:
      {
        extern float flowRateLPM;
        long pLong = map((long)(flowRateLPM * 10), 10, 300, 2000, 400);
        uint32_t per = (uint32_t)constrain(pLong, 400L, 2000L);
        rgbWrite(0, breathe(per, 255), 0);
        break;
      }

    // 2x yellow flashes then dim amber hold
    case LED_REQUEST_RX:
      {
        const uint32_t F = 80;
        switch (ledPhase) {
          case 0:
            rgbWrite(COL_REQ);
            if (elapsed >= F) {
              ledPhase = 1;
              ledPhaseMs = millis();
            }
            break;
          case 1:
            rgbOff();
            if (elapsed >= F) {
              ledPhase = 2;
              ledPhaseMs = millis();
            }
            break;
          case 2:
            rgbWrite(COL_REQ);
            if (elapsed >= F) {
              ledPhase = 3;
              ledPhaseMs = millis();
            }
            break;
          case 3:
            rgbOff();
            if (elapsed >= F) {
              ledPhase = 4;
              ledPhaseMs = millis();
            }
            break;
          case 4: rgbWrite(120, 100, 0); break;  // dim amber hold
        }
        break;
      }

    case LED_TRANSMITTING:
      rgbWrite((elapsed % 100) < 50 ? COL_TX : COL_OFF);
      break;

    case LED_TX_SUCCESS:
      if (elapsed < 300) rgbWrite(COL_OK);
      else enterLedState(ledBaseState);
      break;

    case LED_TX_FAIL:
      {
        const uint32_t F = 100;
        if (ledPhase < 5) {
          rgbWrite((ledPhase % 2 == 0) ? COL_FAIL : COL_OFF);
          if (elapsed >= F) {
            ledPhase++;
            ledPhaseMs = millis();
          }
        } else {
          enterLedState(ledBaseState);
        }
        break;
      }

    case LED_SD_WRITE:
      if (elapsed < 80) rgbWrite(COL_SD);
      else enterLedState(ledBaseState);
      break;

    case LED_HW_ERROR:
      rgbWrite((elapsed % 1000) < 500 ? COL_FAIL : COL_OFF);
      break;
  }
}

// ══════════════════════════════════════════════
//  Application globals
// ══════════════════════════════════════════════
volatile uint32_t pulseCount = 0;
float flowRateLPM = 0.0f;
float totalLitres = 0.0f;
uint32_t lastSampleMs = 0;
uint32_t lastFlowMs = 0;
bool sdReady = false;
bool rtcReady = false;
bool hwError = false;

RTC_DS3231 rtc;
RF24 radio(NRF_CE_PIN, NRF_CSN_PIN);

// ──────────────────────────────────────────────
//  Flow pulse ISR  (INT0 – pin 2)
// ──────────────────────────────────────────────
void flowPulseISR() {
  pulseCount++;
}

// ──────────────────────────────────────────────
//  SD helpers
// ──────────────────────────────────────────────
String buildTimestamp(DateTime &dt) {
  char buf[20];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
           dt.year(), dt.month(), dt.day(),
           dt.hour(), dt.minute(), dt.second());
  return String(buf);
}

String csvFilename(DateTime &dt) {
  char buf[40];
  snprintf(buf, sizeof(buf), "/NODE%d/%04d/%02d/data_%04d%02d%02d.csv",
           NODE_ID, dt.year(), dt.month(), dt.year(), dt.month(), dt.day());
  return String(buf);
}

String txtFilename(DateTime &dt) {
  char buf[40];
  snprintf(buf, sizeof(buf), "/NODE%d/%04d/%02d/log_%04d%02d%02d.txt",
           NODE_ID, dt.year(), dt.month(), dt.year(), dt.month(), dt.day());
  return String(buf);
}

void ensureDir(const String &fp) {
  int ls = fp.lastIndexOf('/');
  if (ls <= 0) return;
  String d = fp.substring(0, ls);
  if (!SD.exists(d)) SD.mkdir(d);
}

void logToCSV(DateTime &dt, float rate, float total) {
  String fn = csvFilename(dt);
  ensureDir(fn);
  bool nf = !SD.exists(fn);
  File f = SD.open(fn, FILE_WRITE);
  if (!f) return;
  if (nf) f.println(F("timestamp,node_id,flow_rate_lpm,total_litres"));
  f.print(buildTimestamp(dt));
  f.print(',');
  f.print(NODE_ID);
  f.print(',');
  f.print(rate, 3);
  f.print(',');
  f.println(total, 3);
  f.close();
}

void logToTXT(DateTime &dt, float rate, float total) {
  String fn = txtFilename(dt);
  ensureDir(fn);
  File f = SD.open(fn, FILE_WRITE);
  if (!f) return;
  f.print(F("["));
  f.print(buildTimestamp(dt));
  f.print(F("] NODE"));
  f.print(NODE_ID);
  f.print(F(" | Rate: "));
  f.print(rate, 3);
  f.print(F(" L/min | Total: "));
  f.print(total, 3);
  f.println(F(" L"));
  f.close();
}

// ──────────────────────────────────────────────
//  Handle an NRF request validated in loop()
// ──────────────────────────────────────────────
void handleMasterRequest() {
  // LED: two yellow flashes
  if (!hwError) ledRequestRx();
  uint32_t waitUntil = millis() + 400;
  while (millis() < waitUntil) ledTick();

  // LED: blue strobe while transmitting
  radio.stopListening();
  if (!hwError) ledTransmitting();

  FlowPacket pkt;
  pkt.nodeId = NODE_ID;
  pkt.timestamp = rtcReady ? rtc.now().unixtime() : 0UL;
  pkt.flowRateLPM = flowRateLPM;
  pkt.totalLitres = totalLitres;
  pkt.status = (uint8_t)((!sdReady ? 1 : 0) | (!rtcReady ? 2 : 0));

  bool ok = radio.write(&pkt, sizeof(pkt));

  // LED: result flash
  if (!hwError) { ok ? ledTxOk() : ledTxFail(); }

  // Resume listening.
  // RF24 v1.5.0 does not have clearWritingPipe().
  // stopListening() + startListening() is the correct sequence.
  radio.startListening();

  // Clear the NRF status register so IRQ line de-asserts,
  // then discard any spurious flag set during the TX window.
  bool txOk, txFail, rxReady;
  radio.whatHappened(txOk, txFail, rxReady);
  nrfIrqFired = false;

  Serial.print(F("TX to master: "));
  Serial.println(ok ? F("OK") : F("FAIL"));
}

// ──────────────────────────────────────────────
//  Setup
// ──────────────────────────────────────────────
void setup() {
  Serial.begin(9600);

  // RGB LED pins
  pinMode(RGB_R_PIN, OUTPUT);
  pinMode(RGB_G_PIN, OUTPUT);
  pinMode(RGB_B_PIN, OUTPUT);
  rgbOff();

  // Startup colour sweep – confirms all three channels work
  RGB sweep[] = {
    { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 0 }, { 0, 255, 255 }, { 255, 0, 255 }, { 255, 255, 255 }
  };
  for (uint8_t i = 0; i < 7; i++) {
    rgbWrite(sweep[i]);
    delay(180);
  }
  rgbOff();
  delay(120);

  // Flow sensor on INT0 (pin 2)
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseISR, RISING);

  // RTC
  Wire.begin();
  if (rtc.begin()) {
    rtcReady = true;
    if (rtc.lostPower()) {
      rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
      Serial.println(F("RTC: compile-time fallback"));
    }
  } else {
    Serial.println(F("ERROR: RTC"));
    hwError = true;
  }

  // SD card
  if (SD.begin(SD_CS_PIN)) {
    sdReady = true;
    char d[12];
    snprintf(d, sizeof(d), "/NODE%d", NODE_ID);
    if (!SD.exists(d)) SD.mkdir(d);
    Serial.println(F("SD: OK"));
  } else {
    Serial.println(F("ERROR: SD"));
    hwError = true;
  }

  // NRF24L01
  if (!radio.begin()) {
    Serial.println(F("ERROR: NRF24L01"));
    hwError = true;
  }
  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(108);
  radio.setPayloadSize(sizeof(FlowPacket));
  radio.setRetries(5, 15);
  radio.openReadingPipe(1, RX_PIPE);
  radio.openWritingPipe(TX_PIPE);

  // Mask TX_DS and MAX_RT on the NRF chip.
  // Only RX_DR (packet received) will assert the IRQ line LOW.
  //   maskIRQ(tx_ok, tx_fail, rx_ready)
  //   true  = masked  (will NOT fire IRQ)
  //   false = enabled (WILL fire IRQ)
  radio.maskIRQ(true, true, false);

  // Arm ATmega PCINT on pin 4 for the NRF IRQ line
  enableNrfIrq();

  radio.startListening();

  if (hwError) ledHwError();
  else ledIdle();

  lastSampleMs = millis();
  Serial.print(F("Node "));
  Serial.print(NODE_ID);
  Serial.println(F(" ready."));
  Serial.println(F("NRF IRQ armed on pin 4 (PCINT20)."));
}

// ──────────────────────────────────────────────
//  Main loop
// ──────────────────────────────────────────────
void loop() {
  uint32_t now = millis();

  // Always tick LED first (non-blocking)
  ledTick();

  // ── 1. Periodic flow-rate calculation ────────────────────
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    float elapsedSec = (float)(now - lastSampleMs) / 1000.0f;

    noInterrupts();
    uint32_t pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    flowRateLPM = ((float)pulses / elapsedSec) / CALIBRATION_FACTOR;
    totalLitres += (flowRateLPM * (float)(now - lastSampleMs)) / 60000.0f;
    lastSampleMs = now;

    if (flowRateLPM > 0.01f) {
      lastFlowMs = now;
      if (!hwError) ledFlowActive();

      if (sdReady) {
        DateTime dt = rtcReady ? rtc.now() : DateTime(0UL);
        if (!hwError) ledSdWrite();
        logToCSV(dt, flowRateLPM, totalLitres);
        logToTXT(dt, flowRateLPM, totalLitres);
      }
    } else {
      // Revert to IDLE after flow stops
      if (!hwError && ledBaseState == LED_FLOW_ACTIVE && (now - lastFlowMs) >= FLOW_IDLE_TIMEOUT_MS) {
        ledIdle();
      }
    }

    Serial.print(F("Flow: "));
    Serial.print(flowRateLPM, 3);
    Serial.print(F(" L/min  Total: "));
    Serial.print(totalLitres, 3);
    Serial.println(F(" L"));
  }

  // ── 2. NRF IRQ – flag set by PCINT2_vect ISR ─────────────
  //  Entered ONLY when the NRF pulls IRQ LOW (packet in FIFO).
  //  No radio.available() polling anywhere in this code.
  if (nrfIrqFired) {
    // Clear flag atomically before touching SPI
    noInterrupts();
    nrfIrqFired = false;
    interrupts();

    // Determine which NRF event fired and clear status register
    bool txOk, txFail, rxReady;
    radio.whatHappened(txOk, txFail, rxReady);

    if (rxReady) {
      uint8_t reqNode = 0;
      radio.read(&reqNode, sizeof(reqNode));

      if (reqNode == NODE_ID) {
        handleMasterRequest();
      } else {
        // Packet addressed to another node – discard
        Serial.print(F("IRQ: ignored (node "));
        Serial.print(reqNode);
        Serial.println(F(")"));
      }
    }
  }
}/**
 * ============================================================
 *  WATER FLOW NODE 2  –  ATmega328P  (v3 – NRF IRQ driven)
 * ============================================================
 *
 *  Pin Assignment  (identical hardware layout to Node 1)
 *  ──────────────────────────────────────────────────────────
 *  Pin  2  (INT0)   <- YF-S201 flow pulse         [interrupt]
 *  Pin  3  (PWM)    -> RGB LED  Red   + 220Ohm
 *  Pin  4           <- NRF24L01 IRQ               [PCINT20]
 *  Pin  5  (PWM)    -> RGB LED  Green + 220Ohm
 *  Pin  6  (PWM)    -> RGB LED  Blue  + 220Ohm
 *  Pin  8           -> SD card  CS
 *  Pin  9           -> NRF24L01 CE
 *  Pin 10           -> NRF24L01 CSN
 *  Pin 11  (MOSI)   -> SPI bus
 *  Pin 12  (MISO)   <- SPI bus
 *  Pin 13  (SCK)    -> SPI bus
 *  Pin A4  (SDA)    <-> DS3231 RTC
 *  Pin A5  (SCL)    <-> DS3231 RTC
 *
 *  NOTE: Only NODE_ID (2) and RX_PIPE differ from Node 1.
 *        All IRQ, LED, and flow logic is identical.
 *
 *  RGB LED States
 *  ──────────────────────────────────────────────────────────
 *  IDLE         Cyan    Slow 3s breath     Listening, no flow
 *  FLOW_ACTIVE  Green   Rate-scaled breath Flow detected
 *  REQUEST_RX   Yellow  2x 80ms flash      IRQ fired
 *  TRANSMITTING Blue    50ms strobe        RF TX active
 *  TX_SUCCESS   White   300ms solid        ACK confirmed
 *  TX_FAIL      Red     3x 100ms flash     No ACK
 *  SD_WRITE     Magenta 80ms blip          Writing SD
 *  HW_ERROR     Red     1Hz blink          Hardware fault
 * ============================================================
 */

// ── Includes ────────────────────────────────────────────────
#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <RTClib.h>
#include <RF24.h>
#include <avr/interrupt.h>

// ── ALL TYPE DEFINITIONS BEFORE ANY FUNCTIONS ───────────────
// Placed here so the Arduino IDE prototype generator sees them
// before generating any forward declarations.

struct RGB {
  uint8_t r, g, b;
};

enum LedState : uint8_t {
  LED_IDLE,
  LED_FLOW_ACTIVE,
  LED_REQUEST_RX,
  LED_TRANSMITTING,
  LED_TX_SUCCESS,
  LED_TX_FAIL,
  LED_SD_WRITE,
  LED_HW_ERROR
};

struct FlowPacket {
  uint8_t nodeId;
  uint32_t timestamp;
  float flowRateLPM;
  float totalLitres;
  uint8_t status;
};

// ──────────────────────────────────────────────
//  Pin definitions
// ──────────────────────────────────────────────
#define FLOW_SENSOR_PIN 2
#define RGB_R_PIN 3
#define NRF_IRQ_PIN 4  // PCINT20
#define RGB_G_PIN 5
#define RGB_B_PIN 6
#define SD_CS_PIN 8
#define NRF_CE_PIN 9
#define NRF_CSN_PIN 10

#define RGB_COMMON_ANODE false

// ──────────────────────────────────────────────
//  Node identity  <- ONLY difference from Node 1
// ──────────────────────────────────────────────
#define NODE_ID 2
const uint64_t RX_PIPE = 0xB2B2B2B2B2LL;
const uint64_t TX_PIPE = 0xB0B0B0B0B0LL;

// ──────────────────────────────────────────────
//  Timing
// ──────────────────────────────────────────────
#define SAMPLE_INTERVAL_MS 500UL
#define CALIBRATION_FACTOR 7.5f
#define FLOW_IDLE_TIMEOUT_MS 8000UL

// ══════════════════════════════════════════════
//  NRF24L01 IRQ – PCINT2 on pin 4 (PD4 = PCINT20)
// ══════════════════════════════════════════════
volatile bool nrfIrqFired = false;

ISR(PCINT2_vect) {
  if (!(PIND & (1 << PD4))) {
    nrfIrqFired = true;
  }
}

void enableNrfIrq() {
  pinMode(NRF_IRQ_PIN, INPUT);
  PCICR |= (1 << PCIE2);
  PCMSK2 |= (1 << PCINT20);
}

// ══════════════════════════════════════════════
//  RGB colour constants
// ══════════════════════════════════════════════
const RGB COL_OFF = { 0, 0, 0 };
const RGB COL_IDLE = { 0, 80, 80 };
const RGB COL_FLOW = { 0, 255, 0 };
const RGB COL_REQ = { 255, 200, 0 };
const RGB COL_TX = { 0, 0, 255 };
const RGB COL_OK = { 255, 255, 255 };
const RGB COL_FAIL = { 255, 0, 0 };
const RGB COL_SD = { 180, 0, 180 };

// ══════════════════════════════════════════════
//  RGB LED driver
// ══════════════════════════════════════════════
void rgbWrite(uint8_t r, uint8_t g, uint8_t b) {
  if (RGB_COMMON_ANODE) {
    r = 255 - r;
    g = 255 - g;
    b = 255 - b;
  }
  analogWrite(RGB_R_PIN, r);
  analogWrite(RGB_G_PIN, g);
  analogWrite(RGB_B_PIN, b);
}

void rgbWrite(RGB c) {
  rgbWrite(c.r, c.g, c.b);
}

void rgbOff() {
  rgbWrite(0, 0, 0);
}

uint8_t breathe(uint32_t periodMs, uint8_t peak) {
  float t = (float)(millis() % periodMs) / (float)periodMs;
  return (uint8_t)((1.0f - cosf(t * 6.28318f)) * 0.5f * (float)peak);
}

// ══════════════════════════════════════════════
//  LED state machine
// ══════════════════════════════════════════════
LedState ledState = LED_IDLE;
LedState ledBaseState = LED_IDLE;
uint32_t ledPhaseMs = 0;
uint8_t ledPhase = 0;

void enterLedState(LedState s) {
  ledState = s;
  ledPhase = 0;
  ledPhaseMs = millis();
}

void ledSetBase(LedState s) {
  ledBaseState = s;
  enterLedState(s);
}

void ledTransient(LedState s) {
  enterLedState(s);
}

void ledIdle() {
  ledSetBase(LED_IDLE);
}
void ledFlowActive() {
  ledSetBase(LED_FLOW_ACTIVE);
}
void ledRequestRx() {
  ledTransient(LED_REQUEST_RX);
}
void ledTransmitting() {
  ledTransient(LED_TRANSMITTING);
}
void ledTxOk() {
  ledTransient(LED_TX_SUCCESS);
}
void ledTxFail() {
  ledTransient(LED_TX_FAIL);
}
void ledSdWrite() {
  ledTransient(LED_SD_WRITE);
}
void ledHwError() {
  ledSetBase(LED_HW_ERROR);
}

void ledTick() {
  uint32_t elapsed = millis() - ledPhaseMs;

  switch (ledState) {

    case LED_IDLE:
      {
        uint8_t v = breathe(3000, 80);
        rgbWrite(0, v, v);
        break;
      }

    case LED_FLOW_ACTIVE:
      {
        extern float flowRateLPM;
        long pLong = map((long)(flowRateLPM * 10), 10, 300, 2000, 400);
        uint32_t per = (uint32_t)constrain(pLong, 400L, 2000L);
        rgbWrite(0, breathe(per, 255), 0);
        break;
      }

    case LED_REQUEST_RX:
      {
        const uint32_t F = 80;
        switch (ledPhase) {
          case 0:
            rgbWrite(COL_REQ);
            if (elapsed >= F) {
              ledPhase = 1;
              ledPhaseMs = millis();
            }
            break;
          case 1:
            rgbOff();
            if (elapsed >= F) {
              ledPhase = 2;
              ledPhaseMs = millis();
            }
            break;
          case 2:
            rgbWrite(COL_REQ);
            if (elapsed >= F) {
              ledPhase = 3;
              ledPhaseMs = millis();
            }
            break;
          case 3:
            rgbOff();
            if (elapsed >= F) {
              ledPhase = 4;
              ledPhaseMs = millis();
            }
            break;
          case 4: rgbWrite(120, 100, 0); break;
        }
        break;
      }

    case LED_TRANSMITTING:
      rgbWrite((elapsed % 100) < 50 ? COL_TX : COL_OFF);
      break;

    case LED_TX_SUCCESS:
      if (elapsed < 300) rgbWrite(COL_OK);
      else enterLedState(ledBaseState);
      break;

    case LED_TX_FAIL:
      {
        const uint32_t F = 100;
        if (ledPhase < 5) {
          rgbWrite((ledPhase % 2 == 0) ? COL_FAIL : COL_OFF);
          if (elapsed >= F) {
            ledPhase++;
            ledPhaseMs = millis();
          }
        } else {
          enterLedState(ledBaseState);
        }
        break;
      }

    case LED_SD_WRITE:
      if (elapsed < 80) rgbWrite(COL_SD);
      else enterLedState(ledBaseState);
      break;

    case LED_HW_ERROR:
      rgbWrite((elapsed % 1000) < 500 ? COL_FAIL : COL_OFF);
      break;
  }
}

// ══════════════════════════════════════════════
//  Application globals
// ══════════════════════════════════════════════
volatile uint32_t pulseCount = 0;
float flowRateLPM = 0.0f;
float totalLitres = 0.0f;
uint32_t lastSampleMs = 0;
uint32_t lastFlowMs = 0;
bool sdReady = false;
bool rtcReady = false;
bool hwError = false;

RTC_DS3231 rtc;
RF24 radio(NRF_CE_PIN, NRF_CSN_PIN);

void flowPulseISR() {
  pulseCount++;
}

// ──────────────────────────────────────────────
//  SD helpers
// ──────────────────────────────────────────────
String buildTimestamp(DateTime &dt) {
  char buf[20];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
           dt.year(), dt.month(), dt.day(),
           dt.hour(), dt.minute(), dt.second());
  return String(buf);
}

String csvFilename(DateTime &dt) {
  char buf[40];
  snprintf(buf, sizeof(buf), "/NODE%d/%04d/%02d/data_%04d%02d%02d.csv",
           NODE_ID, dt.year(), dt.month(), dt.year(), dt.month(), dt.day());
  return String(buf);
}

String txtFilename(DateTime &dt) {
  char buf[40];
  snprintf(buf, sizeof(buf), "/NODE%d/%04d/%02d/log_%04d%02d%02d.txt",
           NODE_ID, dt.year(), dt.month(), dt.year(), dt.month(), dt.day());
  return String(buf);
}

void ensureDir(const String &fp) {
  int ls = fp.lastIndexOf('/');
  if (ls <= 0) return;
  String d = fp.substring(0, ls);
  if (!SD.exists(d)) SD.mkdir(d);
}

void logToCSV(DateTime &dt, float rate, float total) {
  String fn = csvFilename(dt);
  ensureDir(fn);
  bool nf = !SD.exists(fn);
  File f = SD.open(fn, FILE_WRITE);
  if (!f) return;
  if (nf) f.println(F("timestamp,node_id,flow_rate_lpm,total_litres"));
  f.print(buildTimestamp(dt));
  f.print(',');
  f.print(NODE_ID);
  f.print(',');
  f.print(rate, 3);
  f.print(',');
  f.println(total, 3);
  f.close();
}

void logToTXT(DateTime &dt, float rate, float total) {
  String fn = txtFilename(dt);
  ensureDir(fn);
  File f = SD.open(fn, FILE_WRITE);
  if (!f) return;
  f.print(F("["));
  f.print(buildTimestamp(dt));
  f.print(F("] NODE"));
  f.print(NODE_ID);
  f.print(F(" | Rate: "));
  f.print(rate, 3);
  f.print(F(" L/min | Total: "));
  f.print(total, 3);
  f.println(F(" L"));
  f.close();
}

// ──────────────────────────────────────────────
//  Handle a validated NRF request
// ──────────────────────────────────────────────
void handleMasterRequest() {
  if (!hwError) ledRequestRx();
  uint32_t waitUntil = millis() + 400;
  while (millis() < waitUntil) ledTick();

  radio.stopListening();
  if (!hwError) ledTransmitting();

  FlowPacket pkt;
  pkt.nodeId = NODE_ID;
  pkt.timestamp = rtcReady ? rtc.now().unixtime() : 0UL;
  pkt.flowRateLPM = flowRateLPM;
  pkt.totalLitres = totalLitres;
  pkt.status = (uint8_t)((!sdReady ? 1 : 0) | (!rtcReady ? 2 : 0));

  bool ok = radio.write(&pkt, sizeof(pkt));

  if (!hwError) { ok ? ledTxOk() : ledTxFail(); }

  // RF24 v1.5.0: stopListening() then startListening() is correct.
  // clearWritingPipe() does not exist in v1.5.0.
  radio.startListening();

  bool txOk, txFail, rxReady;
  radio.whatHappened(txOk, txFail, rxReady);
  nrfIrqFired = false;

  Serial.print(F("TX to master: "));
  Serial.println(ok ? F("OK") : F("FAIL"));
}

// ──────────────────────────────────────────────
//  Setup
// ──────────────────────────────────────────────
void setup() {
  Serial.begin(9600);

  pinMode(RGB_R_PIN, OUTPUT);
  pinMode(RGB_G_PIN, OUTPUT);
  pinMode(RGB_B_PIN, OUTPUT);
  rgbOff();

  RGB sweep[] = {
    { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 0 }, { 0, 255, 255 }, { 255, 0, 255 }, { 255, 255, 255 }
  };
  for (uint8_t i = 0; i < 7; i++) {
    rgbWrite(sweep[i]);
    delay(180);
  }
  rgbOff();
  delay(120);

  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseISR, RISING);

  Wire.begin();
  if (rtc.begin()) {
    rtcReady = true;
    if (rtc.lostPower()) {
      rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
      Serial.println(F("RTC: compile-time fallback"));
    }
  } else {
    Serial.println(F("ERROR: RTC"));
    hwError = true;
  }

  if (SD.begin(SD_CS_PIN)) {
    sdReady = true;
    char d[12];
    snprintf(d, sizeof(d), "/NODE%d", NODE_ID);
    if (!SD.exists(d)) SD.mkdir(d);
    Serial.println(F("SD: OK"));
  } else {
    Serial.println(F("ERROR: SD"));
    hwError = true;
  }

  if (!radio.begin()) {
    Serial.println(F("ERROR: NRF24L01"));
    hwError = true;
  }
  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(108);
  radio.setPayloadSize(sizeof(FlowPacket));
  radio.setRetries(5, 15);
  radio.openReadingPipe(1, RX_PIPE);
  radio.openWritingPipe(TX_PIPE);
  radio.maskIRQ(true, true, false);  // only RX_DR fires the IRQ pin
  enableNrfIrq();
  radio.startListening();

  if (hwError) ledHwError();
  else ledIdle();

  lastSampleMs = millis();
  Serial.print(F("Node "));
  Serial.print(NODE_ID);
  Serial.println(F(" ready."));
  Serial.println(F("NRF IRQ armed on pin 4 (PCINT20)."));
}

// ──────────────────────────────────────────────
//  Main loop
// ──────────────────────────────────────────────
void loop() {
  uint32_t now = millis();

  ledTick();

  // ── 1. Flow-rate calculation ──────────────────────────────
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    float elapsedSec = (float)(now - lastSampleMs) / 1000.0f;

    noInterrupts();
    uint32_t pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    flowRateLPM = ((float)pulses / elapsedSec) / CALIBRATION_FACTOR;
    totalLitres += (flowRateLPM * (float)(now - lastSampleMs)) / 60000.0f;
    lastSampleMs = now;

    if (flowRateLPM > 0.01f) {
      lastFlowMs = now;
      if (!hwError) ledFlowActive();
      if (sdReady) {
        DateTime dt = rtcReady ? rtc.now() : DateTime(0UL);
        if (!hwError) ledSdWrite();
        logToCSV(dt, flowRateLPM, totalLitres);
        logToTXT(dt, flowRateLPM, totalLitres);
      }
    } else {
      if (!hwError && ledBaseState == LED_FLOW_ACTIVE && (now - lastFlowMs) >= FLOW_IDLE_TIMEOUT_MS) {
        ledIdle();
      }
    }

    Serial.print(F("Flow: "));
    Serial.print(flowRateLPM, 3);
    Serial.print(F(" L/min  Total: "));
    Serial.print(totalLitres, 3);
    Serial.println(F(" L"));
  }

  // ── 2. NRF IRQ – no polling, flag set by PCINT2_vect ─────
  if (nrfIrqFired) {
    noInterrupts();
    nrfIrqFired = false;
    interrupts();

    bool txOk, txFail, rxReady;
    radio.whatHappened(txOk, txFail, rxReady);

    if (rxReady) {
      uint8_t reqNode = 0;
      radio.read(&reqNode, sizeof(reqNode));

      if (reqNode == NODE_ID) {
        handleMasterRequest();
      } else {
        Serial.print(F("IRQ: ignored (node "));
        Serial.print(reqNode);
        Serial.println(F(")"));
      }
    }
  }
}
/*
 * UICT Water Usage Gateway – NWSC Dashboard
 * Title: WATER_USAGE_PREDICTION_SYSTEM
 * Initial values: 3.1 liters for both blocks
 * Random fallback: after 90s, each node regenerates every 102-210 seconds
 * Login users: admin/admin and Owen/@Owenie12
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SPI.h>
#include <SD.h>
#include <RF24.h>
#include <ArduinoJson.h>
#include <time.h>

// --------------------- Pin Definitions ---------------------
#define SD_CS    D1
#define NRF_CE   D0
#define NRF_CSN  D4
#define LCD_ADDR 0x27

LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);
RF24 radio(NRF_CE, NRF_CSN);

const char* ssid     = "Owen";
const char* password = "@Owenie12";
const char* ntpServer      = "africa.pool.ntp.org";
const long  gmtOffset_sec  = 10800;
const int   daylightOffset_sec = 0;

const byte address1[6] = "Node1";
const byte address2[6] = "Node2";

float block1_liters = 3.1;   // initial value
float block2_liters = 3.1;   // initial value
float cost_per_liter = 2.0;

unsigned long lastRxTime1 = 0;
unsigned long lastRxTime2 = 0;
unsigned long lastLogTime = 0;
const unsigned long logInterval = 5000;

// Fallback delay: 90 seconds after startup before any random generation
unsigned long startupTime = 0;
const unsigned long fallbackDelay = 90000;

// Per‑node random regeneration intervals (milliseconds)
unsigned long lastRandomTime1 = 0;
unsigned long lastRandomTime2 = 0;
unsigned long nextRandomInterval1 = 0;
unsigned long nextRandomInterval2 = 0;

#define MIN_RANDOM_INTERVAL 102000   // 1.7 minutes (102 seconds)
#define MAX_RANDOM_INTERVAL 210000   // 3.5 minutes (210 seconds)

#define MAX_HISTORY 30
float total_usage_history[MAX_HISTORY];
int history_index = 0;
int history_count = 0;

ESP8266WebServer server(80);

// --------------------- Authentication (multiple users) ---------------------
struct User {
  const char* username;
  const char* password;
};
User validUsers[] = {
  {"admin", "admin"},
  {"Owen", "@Owenie12"}
};
const int userCount = 2;

String validSessionToken = "";

String generateToken() {
  String token = "";
  for (int i = 0; i < 16; i++) token += char(random(65, 90));
  return token;
}

bool isValidLogin(String user, String pass) {
  for (int i = 0; i < userCount; i++) {
    if (user == validUsers[i].username && pass == validUsers[i].password) {
      return true;
    }
  }
  return false;
}

bool isAuthenticated() {
  if (!server.hasHeader("Cookie")) return false;
  String cookie = server.header("Cookie");
  int idx = cookie.indexOf("session=");
  if (idx == -1) return false;
  int start = idx + 8;
  int end = cookie.indexOf(";", start);
  if (end == -1) end = cookie.length();
  return (cookie.substring(start, end) == validSessionToken && validSessionToken.length() > 0);
}

bool checkAuthAndRedirect() {
  if (!isAuthenticated()) {
    server.sendHeader("Location", "/login", true);
    server.send(302, "text/plain", "");
    return false;
  }
  return true;
}

// --------------------- Helpers ---------------------
String getFormattedTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 5000)) return "N/A";
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buffer);
}

float getRandomFallback() {
  return random(1534, 3601) / 1000.0;
}

void logToSD(float l1, float l2) {
  File dataFile = SD.open("/water_data.csv", FILE_WRITE);
  if (dataFile) {
    dataFile.print(getFormattedTimestamp());
    dataFile.print(","); dataFile.print(l1, 3);
    dataFile.print(","); dataFile.println(l2, 3);
    dataFile.close();
  }
}

float predictTomorrowUsage() {
  if (history_count < 2) return 0;
  float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  int n = history_count;
  for (int i = 0; i < n; i++) {
    float x = i, y = total_usage_history[i];
    sumX += x; sumY += y; sumXY += x * y; sumX2 += x * x;
  }
  float denom = n * sumX2 - sumX * sumX;
  if (denom == 0) return 0;
  float slope = (n * sumXY - sumX * sumY) / denom;
  float intercept = (sumY - slope * sumX) / n;
  float predicted = slope * n + intercept;
  return predicted > 0 ? predicted : 0;
}

String buildDataJSON() {
  DynamicJsonDocument doc(256);
  doc["block1"] = block1_liters;
  doc["block2"] = block2_liters;
  doc["cost_per_liter"] = cost_per_liter;
  doc["cost1"] = block1_liters * cost_per_liter;
  doc["cost2"] = block2_liters * cost_per_liter;
  float pred = predictTomorrowUsage();
  doc["predicted_tomorrow"] = pred;
  doc["predicted_cost_tomorrow"] = pred * cost_per_liter;
  doc["timestamp"] = getFormattedTimestamp();
  String out; serializeJson(doc, out); return out;
}

String buildHistoryJSON() {
  DynamicJsonDocument doc(1024);
  JsonArray arr = doc.to<JsonArray>();
  for (int i = 0; i < history_count; i++) {
    arr.add(total_usage_history[i]);
  }
  String out; serializeJson(doc, out); return out;
}

// --------------------- Web Handlers ---------------------
void handleLogin() {
  if (server.method() == HTTP_POST) {
    String user = server.arg("username");
    String pass = server.arg("password");
    if (isValidLogin(user, pass)) {
      validSessionToken = generateToken();
      server.sendHeader("Set-Cookie", "session=" + validSessionToken + "; path=/");
      server.sendHeader("Location", "/", true);
      server.send(302, "text/plain", "");
      Serial.println("Login OK for: " + user);
    } else {
      server.sendHeader("Location", "/login?err=1", true);
      server.send(302, "text/plain", "");
    }
    return;
  }
  bool err = server.hasArg("err");
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width'><title>WATER_USAGE_PREDICTION_SYSTEM - Login</title><style>body{background:#003087;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif}.box{background:white;padding:40px;border-radius:16px;text-align:center}.err{color:red}</style></head><body><div class='box'><h2>WATER_USAGE_PREDICTION_SYSTEM</h2><p>NWSC Water Monitor</p>";
  if (err) html += "<p class='err'>Invalid credentials</p>";
  html += "<form method='POST'><input name='username' placeholder='Username'><br><input type='password' name='password' placeholder='Password'><br><button type='submit'>Login</button></form></div></body></html>";
  server.send(200, "text/html", html);
}

void handleLogout() {
  validSessionToken = "";
  server.sendHeader("Set-Cookie", "session=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT");
  server.sendHeader("Location", "/login", true);
  server.send(302, "text/plain", "");
}

void handleRoot() {
  if (!checkAuthAndRedirect()) return;
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width'><title>WATER_USAGE_PREDICTION_SYSTEM - Dashboard</title><script src='https://cdn.jsdelivr.net/npm/chart.js'></script><style>body{font-family:sans-serif;background:#F0F4F8;padding:20px}.container{max-width:1200px;margin:auto}.header{background:#003087;color:white;padding:20px;border-radius:20px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{background:white;padding:20px;border-radius:16px;flex:1;text-align:center}.value{font-size:2rem;color:#00843D}.logout-btn,.nav-btn{background:#00843D;padding:8px 20px;border-radius:30px;text-decoration:none;color:white;margin-left:10px}button{background:#00843D;border:none;padding:10px 20px;border-radius:30px;color:white;cursor:pointer}.nav{display:flex}</style></head><body><div class='container'><div class='header'><h2>💧 WATER_USAGE_PREDICTION_SYSTEM</h2><div class='nav'><a href='/charts' class='nav-btn'>📈 Charts</a><a href='/logout' class='logout-btn'>Logout</a></div></div><div class='cards'><div class='card'><h3>Block 1</h3><div class='value' id='b1'>3.10</div><div>liters</div></div><div class='card'><h3>Block 2</h3><div class='value' id='b2'>3.10</div><div>liters</div></div><div class='card'><h3>Cost per Liter</h3><div class='value' id='cpl'>2.00</div><div>UGX</div></div><div class='card'><h3>Block 1 Cost</h3><div class='value' id='c1'>0.00</div><div>UGX</div></div><div class='card'><h3>Block 2 Cost</h3><div class='value' id='c2'>0.00</div><div>UGX</div></div><div class='card'><h3>Raspberry Pi prediction</h3><div class='value' id='pred'>0.00</div><div>total liters tomorrow</div><div id='predCost'>0 UGX</div></div></div><div style='text-align:center;margin-top:20px'><button onclick='downloadCSV()'>📥 Download CSV</button><button onclick='fetchData()'>🔄 Refresh</button></div></div><script>async function fetchData(){let r=await fetch('/data');if(!r.ok){if(r.status===302||r.status===401)window.location.href='/login';return;}let d=await r.json();document.getElementById('b1').innerText=d.block1.toFixed(2);document.getElementById('b2').innerText=d.block2.toFixed(2);document.getElementById('cpl').innerText=d.cost_per_liter.toFixed(2);document.getElementById('c1').innerText=d.cost1.toFixed(2);document.getElementById('c2').innerText=d.cost2.toFixed(2);document.getElementById('pred').innerText=d.predicted_tomorrow.toFixed(2);document.getElementById('predCost').innerHTML=d.predicted_cost_tomorrow.toFixed(2)+' UGX';}function downloadCSV(){window.location.href='/download';}fetchData();setInterval(fetchData,5000);</script></body></html>";
  server.send(200, "text/html", html);
}

void handleCharts() {
  if (!checkAuthAndRedirect()) return;
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width'><title>WATER_USAGE_PREDICTION_SYSTEM - Historical Charts</title><script src='https://cdn.jsdelivr.net/npm/chart.js'></script><style>body{font-family:sans-serif;background:#F0F4F8;padding:20px}.container{max-width:1200px;margin:auto}.header{background:#003087;color:white;padding:20px;border-radius:20px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}.chart-box{background:white;border-radius:16px;padding:20px;margin-bottom:20px}.back-btn{background:#00843D;padding:8px 20px;border-radius:30px;text-decoration:none;color:white}</style></head><body><div class='container'><div class='header'><h2>📊 Historical Water Usage (Last 30 readings)</h2><a href='/' class='back-btn'>← Back to Dashboard</a></div><div class='chart-box'><canvas id='historyChart' width='800' height='400'></canvas></div></div><script>async function loadHistory(){let r=await fetch('/history');if(!r.ok)return;let data=await r.json();const ctx=document.getElementById('historyChart').getContext('2d');new Chart(ctx,{type:'line',data:{labels:data.map((_,i)=>i+1),datasets:[{label:'Total Water Usage (liters)',data:data,borderColor:'#003087',backgroundColor:'rgba(0,48,135,0.1)',fill:true,tension:0.3}]},options:{responsive:true,scales:{y:{beginAtZero:true,title:{display:true,text:'Liters'}},x:{title:{display:true,text:'Sample Number'}}}}})}loadHistory();</script></body></html>";
  server.send(200, "text/html", html);
}

void handleData() {
  if (!checkAuthAndRedirect()) return;
  server.send(200, "application/json", buildDataJSON());
}

void handleHistory() {
  if (!checkAuthAndRedirect()) return;
  server.send(200, "application/json", buildHistoryJSON());
}

void handleDownload() {
  if (!checkAuthAndRedirect()) return;
  File file = SD.open("/water_data.csv", FILE_READ);
  if (!file) { server.send(404, "text/plain", "No data"); return; }
  server.streamFile(file, "text/csv");
  file.close();
}

void setupWebServer() {
  server.collectHeaders("Cookie");
  server.on("/login", HTTP_GET, handleLogin);
  server.on("/login", HTTP_POST, handleLogin);
  server.on("/logout", handleLogout);
  server.on("/", handleRoot);
  server.on("/charts", handleCharts);
  server.on("/data", handleData);
  server.on("/history", handleHistory);
  server.on("/download", handleDownload);
  server.begin();
  Serial.println("HTTP server started");
}

// --------------------- Setup ---------------------
void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
  validSessionToken = "";
  startupTime = millis();

  // Set initial random intervals (will be used after fallback starts)
  nextRandomInterval1 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
  nextRandomInterval2 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
  lastRandomTime1 = millis();
  lastRandomTime2 = millis();

  Wire.begin(D2, D3);
  lcd.init(); lcd.backlight(); lcd.print("Gateway v9.0");
  delay(1000);

  WiFi.begin(ssid, password);
  lcd.clear(); lcd.print("WiFi...");
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.println("IP: " + WiFi.localIP().toString());
  lcd.clear(); lcd.print(WiFi.localIP());

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  lcd.setCursor(0,1); lcd.print("Time sync");
  struct tm ti; int tries=0;
  while (!getLocalTime(&ti,5000) && tries++<6) delay(500);
  delay(1500);

  SPI.begin();
  if (SD.begin(SD_CS)) {
    if (!SD.exists("/water_data.csv")) {
      File f = SD.open("/water_data.csv", FILE_WRITE);
      if (f) { f.println("Timestamp,Block1_Liters,Block2_Liters"); f.close(); }
    }
  }

  if (radio.begin()) {
    radio.setChannel(100);
    radio.setPALevel(RF24_PA_LOW);
    radio.openReadingPipe(1, address1);
    radio.openReadingPipe(2, address2);
    radio.startListening();
  }

  for (int i=0; i<MAX_HISTORY; i++) total_usage_history[i]=0;
  setupWebServer();
  lcd.clear(); lcd.print("Ready");
  lcd.setCursor(0,1); lcd.print(WiFi.localIP());
}

// --------------------- Loop ---------------------
void loop() {
  server.handleClient();
  unsigned long now = millis();

  if (radio.available()) {
    uint8_t pipe;
    radio.available(&pipe);
    float val;
    radio.read(&val, sizeof(val));
    if (pipe == 1) {
      block1_liters = val;
      lastRxTime1 = now;
      // Reset random timer for this node (real data arrived)
      lastRandomTime1 = now;
      nextRandomInterval1 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
    }
    if (pipe == 2) {
      block2_liters = val;
      lastRxTime2 = now;
      lastRandomTime2 = now;
      nextRandomInterval2 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
    }
  }

  // Fallback random value generation (only after startup delay)
  if (now - startupTime >= fallbackDelay) {
    // Node1: if no recent data and random interval elapsed
    if (now - lastRxTime1 > 7000 && (now - lastRandomTime1) >= nextRandomInterval1) {
      block1_liters = getRandomFallback();
      lastRandomTime1 = now;
      nextRandomInterval1 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
      Serial.println("Node1 random update");
    }
    // Node2: same logic
    if (now - lastRxTime2 > 7000 && (now - lastRandomTime2) >= nextRandomInterval2) {
      block2_liters = getRandomFallback();
      lastRandomTime2 = now;
      nextRandomInterval2 = random(MIN_RANDOM_INTERVAL, MAX_RANDOM_INTERVAL);
      Serial.println("Node2 random update");
    }
  }

  if (now - lastLogTime >= logInterval) {
    lastLogTime = now;
    logToSD(block1_liters, block2_liters);

    float total = block1_liters + block2_liters;
    total_usage_history[history_index] = total;
    history_index = (history_index+1) % MAX_HISTORY;
    if (history_count < MAX_HISTORY) history_count++;

    lcd.clear();
    lcd.setCursor(0,0); lcd.print("B1:"); lcd.print(block1_liters,1);
    lcd.print(" B2:"); lcd.print(block2_liters,1);
    lcd.setCursor(0,1); lcd.print("C1:"); lcd.print(block1_liters*cost_per_liter,0);
    lcd.print(" C2:"); lcd.print(block2_liters*cost_per_liter,0);
  }
}# Group-40-engineering-department-
