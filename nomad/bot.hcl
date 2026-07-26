job "mountain-labeler" {
  datacenters = ["dc1"]
  type        = "service"

  group "bot" {
    count = 1

    restart {
      attempts = 3
      interval = "5m"
      delay    = "25s"
      mode     = "delay"
    }

    task "discord-bot" {
      driver = "raw_exec"

      env {
        PATH = "/Users/tommydoerr/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
      }

      config {
        command = "/bin/bash"
        args = [
          "-c",
          "cd /Users/tommydoerr/dev/is-the-mountain-out && set -a && source cf.env && set +a && uv run --group bot bot run --config mountain.toml"
        ]
      }

      resources {
        cpu    = 200
        memory = 256
      }
    }
  }
}
