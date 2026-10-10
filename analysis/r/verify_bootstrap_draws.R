args <- commandArgs(TRUE)
runtime_arg <- grep('^--file=',commandArgs(),value=TRUE)[1]
root <- dirname(sub('^--file=','',runtime_arg))
source(file.path(root,'runtime_profile.R'))
configure_locked_runtime(file.path(root,'runtime-versions.lock.json'),'primary')
job <- jsonlite::fromJSON(args[1],simplifyVector=FALSE)
participants <- unlist(job$participants,use.names=FALSE)
for (outcome in names(job$outcomes)) {
  record <- job$outcomes[[outcome]]
  .Random.seed <- as.integer(unlist(record$initial_rng_state))
  hashes <- character(record$requested)
  for (i in seq_len(record$requested))
    hashes[i] <- digest::digest(sample(participants,length(participants),replace=TRUE),algo='sha256')
  stopifnot(identical(digest::digest(hashes,algo='sha256'),record$draw_sha256),
            identical(.Random.seed,as.integer(unlist(record$final_rng_state))))
}
cat('BOOTSTRAP_DRAW_VERIFIED outcomes=',length(job$outcomes),'\n',sep='')
