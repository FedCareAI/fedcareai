# Public library and private platform

Publish the contents of the `fedcareai` directory as its own public repository.
It includes local computation, data contracts, the CLI, an optional HTTP client,
tests, synthetic examples, and documentation under GPL-3.0-only.

Keep `fedcareai-platform` in a separate private repository. The frontend,
application backend, account and project management, database, hosted round
coordination, and deployment belong to that repository. Original platform
additions are proprietary; inherited template code retains its MIT license.

The platform runtime does not import the GPL library. The participant CLI is a
separate program that sends JSON and numeric weight files through HTTP. The
platform implements its own numeric archive handling and averaging. Maintain
that process boundary; do not copy GPL implementation code into proprietary
platform modules or add a runtime dependency on the GPL library without
reassessing the licensing of the combined work.

GPL permits commercial use and requires applicable source and license terms
when covered software is distributed. Merely putting programs in different
folders or processes does not conclusively settle whether they are independent
works; the nature of their communication matters. This layout is an engineering
boundary, not a guarantee about every future integration. See the GNU project's
[aggregate guidance](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation)
and [proprietary system guidance](https://www.gnu.org/licenses/gpl-faq.html#GPLInProprietarySystem).

## Release checklist

1. Publish only the library directory, never the shared parent directory.
2. Build the library wheel and source distribution from its own `pyproject.toml`.
3. Include LICENSE and NOTICE in every library release and source download.
4. Keep local CSVs, models, tokens, virtual environments, and `.env` files out of releases.
5. Run library tests independently; run platform integration tests privately.
6. Rebuild the platform's downloadable library archive after library changes.

The private packaging script currently expects the two checkouts to be siblings.
This is a release-time source lookup, not a library runtime dependency. Library
installation and standalone tests do not require the platform checkout.
