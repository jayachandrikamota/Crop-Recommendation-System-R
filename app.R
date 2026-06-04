library(shiny)
library(shinythemes)
library(ggplot2)

# ==========================================================
# LOAD DATA & MODELS
# ==========================================================

data_obj <- readRDS("data/processed_data.rds")
model_obj <- readRDS("models/models.rds")

preproc <- data_obj$preproc
rf_model <- model_obj$rf
dt_model <- model_obj$dt

# ==========================================================
# UI DESIGN
# ==========================================================

ui <- fluidPage(
  theme = shinytheme("flatly"),

  titlePanel("🌾 Smart Crop Recommendation System"),

  fluidRow(

    # ---------------- INPUT PANEL ----------------
    column(4,
           wellPanel(
             h3("🌱 Input Parameters"),

             sliderInput("N","Nitrogen (N)",0,150,90),
             sliderInput("P","Phosphorus (P)",0,150,40),
             sliderInput("K","Potassium (K)",0,150,40),

             sliderInput("temp","Temperature (°C)",0,50,25),
             sliderInput("hum","Humidity (%)",0,100,80),
             sliderInput("ph","Soil pH",0,14,6.5),
             sliderInput("rain","Rainfall (mm)",0,300,200),

             selectInput("model","Select Model",
                         c("Random Forest","Decision Tree")),

             actionButton("predict","🚀 Predict Crop", class="btn-primary")
           )
    ),

    # ---------------- OUTPUT PANEL ----------------
    column(8,

           # Prediction Result
           wellPanel(
             h2("🌾 Recommended Crop"),
             h1(textOutput("result"), style="color:green; font-weight:bold;"),

             h4("Prediction Confidence"),
             tableOutput("prob_table")
           ),

           # Graphs
           fluidRow(
             column(6,
                    wellPanel(
                      h4("🌡 Temperature vs Rainfall"),
                      plotOutput("tempPlot")
                    )
             ),
             column(6,
                    wellPanel(
                      h4("💧 Humidity vs pH"),
                      plotOutput("humPlot")
                    )
             )
           ),

           # Feature Importance
           wellPanel(
             h4("📊 Feature Importance"),
             plotOutput("importancePlot")
           ),

           # Model Comparison
           wellPanel(
             h4("📈 Model Comparison (Accuracy)"),
             plotOutput("comparisonPlot")
           )
    )
  )
)

# ==========================================================
# SERVER LOGIC
# ==========================================================

server <- function(input, output){

  # ---------------- INPUT VALIDATION ----------------
  validate_input <- function(){
    if(input$ph < 0 || input$ph > 14){
      return("Invalid pH value (0–14 allowed)")
    }
    return(NULL)
  }

  # ---------------- PREDICTION ----------------
  observeEvent(input$predict,{

    error_msg <- validate_input()

    if(!is.null(error_msg)){
      output$result <- renderText(error_msg)
      return()
    }

    new_data <- data.frame(
      N=input$N,P=input$P,K=input$K,
      temperature=input$temp,
      humidity=input$hum,
      ph=input$ph,
      rainfall=input$rain
    )

    new_scaled <- predict(preproc, new_data)

    if(input$model=="Decision Tree"){
      pred <- predict(dt_model, new_scaled, type="class")
    } else {
      pred <- predict(rf_model, new_scaled)
    }

    # Output prediction
    output$result <- renderText({
      paste(pred)
    })

    # ---------------- PROBABILITY OUTPUT ----------------
    output$prob_table <- renderTable({
      prob <- predict(rf_model, new_scaled, type="prob")
      round(prob, 3)
    })
  })

  # ---------------- GRAPH 1 ----------------
  output$tempPlot <- renderPlot({
    df <- data.frame(
      temperature=input$temp,
      rainfall=input$rain
    )

    ggplot(df, aes(x=temperature, y=rainfall)) +
      geom_point(size=5, color="blue") +
      ggtitle("Temperature vs Rainfall")
  })

  # ---------------- GRAPH 2 ----------------
  output$humPlot <- renderPlot({
    df <- data.frame(
      humidity=input$hum,
      ph=input$ph
    )

    ggplot(df, aes(x=humidity, y=ph)) +
      geom_point(size=5, color="darkgreen") +
      ggtitle("Humidity vs pH")
  })

  # ---------------- FEATURE IMPORTANCE ----------------
  output$importancePlot <- renderPlot({
    importance_vals <- importance(rf_model)

    barplot(importance_vals[,1],
            main="Feature Importance",
            col="orange",
            las=2)
  })

  # ---------------- MODEL COMPARISON ----------------
  output$comparisonPlot <- renderPlot({

    # Dummy values (replace with real if stored)
    acc_df <- data.frame(
      Model=c("Decision Tree","Random Forest"),
      Accuracy=c(0.85, 0.92)
    )

    ggplot(acc_df, aes(x=Model, y=Accuracy, fill=Model)) +
      geom_bar(stat="identity") +
      ggtitle("Model Accuracy Comparison")
  })
}

# ==========================================================
# RUN APP
# ==========================================================

shinyApp(ui, server)