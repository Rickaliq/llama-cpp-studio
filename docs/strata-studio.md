# Strata profiles in Studio

Studio treats Strata as its own profile-bundle engine. A model record must use
`format: "strata_bundle"`, `artifact.format: "strata_bundle"`,
`artifact.package_kind: "prepared_bundle"`, `compatible_engines: ["strata"]`,
and `local_path` pointing to the reviewed run-config JSON. The JSON continues to
reference the pack, tokenizer, shards, expert profile, runtime, and library
directories; Studio does not copy or modify that source profile.

The adapter supports the reviewed HIP profile shape only. It rejects unknown
profile fields, credentials, non-loopback hosts, non-offline mode, and worker
options outside its explicit allowlist. Studio creates a separate private
server JSON artifact, overrides the HTTP listener to loopback with Studio's
allocated port, and starts the server eagerly. Registry readiness requires
both HTTP success and `loaded: true` from `/health`.

To reduce GPU pressure, `--expert-cache auto` is compiled as 5000 slots,
explicit smaller slot counts are preserved, and `--vram-reserve-mib` is raised
to at least 3072 MiB. These values affect only the generated runtime config;
the saved profile remains unchanged. Other Strata controls are not exposed as
Studio settings yet. The external Python environment must be registered as the
Strata engine's active `binary_path`.

This support has compiler and process-readiness tests only. It does not claim a
Studio proxied inference test or suitability for all Strata profiles/models.
