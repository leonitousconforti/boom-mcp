{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems =
        function:
        nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (
          system: function nixpkgs.legacyPackages.${system}
        );
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          # uv brings its own Python (3.11, like the school environment);
          # node is only for the MCP Inspector.
          packages = with pkgs; [
            uv
            nodejs_22
          ];
          shellHook = ''
            unset PYTHONPATH
            uv sync --all-extras --quiet
            source .venv/bin/activate
          '';
        };
      });
    };
}
