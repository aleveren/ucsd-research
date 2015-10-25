#!/usr/bin/Rscript

baseUrl <- "http://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx"

filename <- "msl_ccam_obs.csv"
localFile <- paste0("data/", filename)
if (!file.exists(localFile)) {
  url <- paste0(baseUrl, "/document/", filename)
  download.file(url, localFile)
}
df <- read.csv(localFile, stringsAsFactors = FALSE)
print(nrow(df))

indices <- which(df$EDR.Type == "CL5" & df$PDS. != "No")
df <- df[indices, , drop=FALSE]
print(nrow(df))

for (detailType in c("RDR", "CCS")) {
  N <- 3 # TODO nrow(df)
  results <- data.frame(
    EDR.Filename = rep("", N),
    Sol = rep(NA, N),
    Detail.Filename = rep("", N),
    Download.Time = rep(NA, N),
    Load.Time = rep(NA, N),
    File.Num.Rows = rep(NA, N),
    File.Num.Cols = rep(NA, N),
    stringsAsFactors = FALSE)

  accumData <- data.frame()

  for (rowIndex in 1:N) {
    cat(paste0("Row ", rowIndex, " of ", N, "\n"))

    row <- df[rowIndex, ]

    detailFilename <- gsub("M1.DAT", "P3.CSV",
        gsub("EDR", detailType, row$EDR.Filename))
    paddedSol <- sprintf("%05d", row$Sol)
    toDownload <- paste0(baseUrl, "/data/sol", paddedSol, "/", detailFilename)
    localFile <- paste0("data/", detailFilename)

    results[rowIndex, "EDR.Filename"] <- row$EDR.Filename
    results[rowIndex, "Sol"] <- row$Sol
    results[rowIndex, "Detail.Filename"] <- detailFilename

    failed <- FALSE
    tryCatch({
      elapsed <- system.time(download.file(toDownload, localFile))
    }, error = function(e) {
      failed <<- TRUE
      cat(paste0("Failed to download ", toDownload, ", continuing anyway\n"))
    })
    if (!failed) {
      results[rowIndex, "Download.Time"] <- elapsed[[3]]
      elapsed <- system.time({
        detailData <- read.csv(localFile, skip = 16)
      })

      # Delete detail files to save space
      file.remove(localFile)
    }

    results[rowIndex, "Load.Time"] <- elapsed[[3]]
    results[rowIndex, "File.Num.Rows"] <- nrow(detailData)
    results[rowIndex, "File.Num.Cols"] <- ncol(detailData)

    wavelengths <- detailData[, 1]
    detailData[, 1] <- NULL
    shotNames <- colnames(detailData)
    detailData <- t(detailData)
    colnames(detailData) <- paste0("Wave.", wavelengths)
    rownames(detailData) <- NULL

    shotNumbers <- as.integer(gsub("shot", "", shotNames))

    detailData <- data.frame(
        Detail.Filename = detailFilename,
        Sol = row$Sol,
        Shot = shotNumbers,
        detailData,
        stringsAsFactors = FALSE)
    detailData <- detailData[shotNames != "median" & shotNames != "mean", ]

    print(dim(detailData))

    accumData <- rbind(accumData, detailData)
  }
  print(dim(accumData))
  print(results)

  downloadStatsFile <- paste0("data/downloadStats", detailType, ".csv")
  write.csv(results, downloadStatsFile, row.names = FALSE)

  accumFile <- paste0("data/accumData", detailType, ".csv.bz2")
  write.csv(accumData, file = pipe(paste0("bzip2 -c > ", accumFile)),
      row.names = FALSE)
  # NOTE: to view this file, use a command like the following:
  # bzip2 -c -d data/accumDataRDR.csv.bz2 | less -S
}

mocFiles <- c(
  "moc_0000_0179.csv",
  "moc_0180_0269.csv",
  "moc_0270_0359.csv",
  "moc_0360_0449.csv",
  "moc_0450_0583.csv",
  "moc_0584_0707.csv",
  "moc_0708_0804.csv",
  "moc_0805_0938.csv"
)

for (f in mocFiles) {
  localFile <- paste0("data/", f)
  if (!file.exists(localFile)) {
    toDownload <- paste0(baseUrl, "/data/moc/", f)
    download.file(toDownload, localFile)
  }
  #df <- read.csv(localFile, skip = 6)
  #print(nrow(df))
}
