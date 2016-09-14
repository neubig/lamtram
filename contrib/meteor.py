#!/usr/bin/env python

# This is an example of an external evaluation measure for Lamtram.  To use
# Meteor as a training target, specify:
# --eval_meas extern:eos=false,run=/path/to/lamtram/contrib/meteor.py
#
# For external scoring, Lamtram opens the specified "run" executable as a sub-
# process and sends one output/reference pair at a time to score.  Lamtram sends
# a line to stdin in the form
# system output ||| reference translation
# and reads a line from stdout that should contain a single float for the score.

import os
import subprocess
import sys

METEOR_URL = "https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz"
METEOR_TGZ = "meteor-1.5.tar.gz"
METEOR_DIR = "meteor-1.5"
METEOR_JAR = "meteor-1.5.jar"

# To use a different version of Meteor, make a copy of this script in the same
# directory and change the args line below.  Text from Lamtram is already
# tokenized, so only apply lower-casing.  Things to try:
# Different languages: -l <de/es/fr/...>
# Different tasks: -t <rank/adq/hter>
# Run `java -jar meteor-1.5.jar` for a full list of options

METEOR_ARGS = "-lower -l en -t tune"


def main():

  # Meteor checkout lives in git-ignored files subdir
  files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
  meteor_jar = os.path.join(files_dir, METEOR_DIR, METEOR_JAR)

  # Download Meteor if not present
  if not os.path.exists(meteor_jar):
    meteor_tgz = os.path.join(files_dir, METEOR_TGZ)
    subprocess.call("wget -c -O {} {}".format(meteor_tgz, METEOR_URL),
                    shell=True)
    subprocess.call("tar -x -f {} -C {}".format(meteor_tgz, files_dir),
                    shell=True)

  # Run Meteor in streaming mode
  meteor = subprocess.Popen(
      "java -Xmx2G -jar {} - - -stdio {}".format(meteor_jar, METEOR_ARGS),
      shell=True,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE)

  # For each line from Lamtram:
  while True:
    line = sys.stdin.readline()
    if not line:
      break
    # Parse the line: out ||| ref
    out, ref = (f.strip() for f in line.split("|||"))
    # Write to Meteor: SCORE ||| ref ||| out
    meteor.stdin.write("SCORE ||| {} ||| {}\n".format(ref, out))
    # Read stats from Meteor
    stats = meteor.stdout.readline()  # still ends with \n
    # Write to Meteor: EVAL ||| stats
    meteor.stdin.write("EVAL ||| {}".format(stats))
    # Read score from Meteor
    score = meteor.stdout.readline()  # still ends with \n
    # Write score to Lamtram
    sys.stdout.write(score)
    # Must flush after every line to avoid process communication deadlock
    sys.stdout.flush()

  # Cleanup
  meteor.stdin.close()
  meteor.wait()


if __name__ == "__main__":
  main()
