# Wikimedia external diagnostic attribution

Audit date: 2026-08-26

This file covers the 26 manually retained images in the external Wikimedia diagnostic. It is an attribution and provenance record, not a single license for the collection. Each image retains its own source terms.

The complete download registry, including source and normalized hashes, is [`dataset_registry.csv`](dataset_registry.csv). The manual inclusion decision is in [`wikimedia_external_manifest.csv`](wikimedia_external_manifest.csv).

## Changes made to every local copy

The project downloaded a Wikimedia Commons source or thumbnail and then:

1. applied the stored EXIF orientation;
2. converted the image to RGB;
3. scaled it to fit within 768 by 768 pixels; and
4. re-encoded it as an optimized JPEG at quality 91.

The local copy is therefore a normalized and re-encoded version of the linked source. The registry preserves both the downloaded-file SHA-256 and the local normalized-file SHA-256.

## Per-license obligations

- **Public domain:** Attribution may not be legally required in every jurisdiction, but the listed creator and source must be retained for provenance.
- **CC BY 4.0:** Credit the creator, link the source and license, retain supplied notices, and identify the normalization changes. License: https://creativecommons.org/licenses/by/4.0/
- **CC BY-SA 4.0:** Meet the CC BY 4.0 conditions and apply a permitted ShareAlike license to adaptations. License: https://creativecommons.org/licenses/by-sa/4.0/
- **CC BY-SA 3.0:** Credit the creator, identify the license, link the source where practicable, identify changes, and preserve ShareAlike for adaptations. License: https://creativecommons.org/licenses/by-sa/3.0/
- **CC BY-SA 2.5:** Credit the creator, identify the license, identify changes, and preserve ShareAlike for adaptations. License: https://creativecommons.org/licenses/by-sa/2.5/
- **CC BY-SA 2.0 France:** Credit the creator, identify the license, identify changes, and preserve ShareAlike for adaptations under the France 2.0 terms. License: https://creativecommons.org/licenses/by-sa/2.0/fr/

Creative Commons licenses do not grant trademark, privacy, publicity, personality, patent, or endorsement rights. Wikimedia also provides no warranty that every uploader's license assertion is correct. Reusers must review the linked file page and applicable non-copyright restrictions.

## Retained files

| Local normalized file | Source title | Creator or attribution party | Recorded terms | Source page |
|---|---|---|---|---|
| `data/wikimedia_pilot/train/safe/safe_train_9d2e2ef3fe4a.jpg` | Baking powder advertisement 1885.jpg | Unknown. Publisher: Buffalo, N.Y. : G.H. Dunston, Lith. | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Baking_powder_advertisement_1885.jpg) |
| `data/wikimedia_pilot/train/safe/safe_train_617511ad1a2f.jpg` | Chiclets advertisement, 1905.jpg | Unknown author | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Chiclets_advertisement,_1905.jpg) |
| `data/wikimedia_pilot/train/safe/safe_train_eb77534129f9.jpg` | Maggi advertisement 1903.jpg | Unknown author | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Maggi_advertisement_1903.jpg) |
| `data/wikimedia_pilot/train/safe/safe_train_c456321d7b18.jpg` | The Shoe for '96.jpg | McClure's Magazine | Public domain | [Source](https://commons.wikimedia.org/wiki/File:The_Shoe_for_%2796.jpg) |
| `data/wikimedia_pilot/val/safe/safe_val_8b0372d1ceac.jpg` | West midlands travel bus 26l07.JPG | Snowmanradio | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [Source](https://commons.wikimedia.org/wiki/File:West_midlands_travel_bus_26l07.JPG) |
| `data/wikimedia_pilot/val/safe/safe_val_b8676660a3e4.jpg` | Manchuria and Korea travel advertisement in 1930s.jpg | Unknown author | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Manchuria_and_Korea_travel_advertisement_in_1930s.jpg) |
| `data/wikimedia_pilot/test/safe/safe_test_5f8657d82e0b.jpg` | Flanders automobile advertisement (from Netherlands).jpg | Unknown author | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Flanders_automobile_advertisement_%28from_Netherlands%29.jpg) |
| `data/wikimedia_pilot/train/firearms/firearms_train_01b73dfab038.jpg` | Palm protector pistol.jpg | Chicago firearms company | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Palm_protector_pistol.jpg) |
| `data/wikimedia_pilot/train/firearms/firearms_train_83fbf1b12e72.jpg` | Browning1900russia.JPG | Unknown author | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Browning1900russia.JPG) |
| `data/wikimedia_pilot/train/firearms/firearms_train_a0c6005948ed.jpg` | .405WinchesterCenterFire.jpg | Townsend Whelen | Public domain | [Source](https://commons.wikimedia.org/wiki/File:.405WinchesterCenterFire.jpg) |
| `data/wikimedia_pilot/train/firearms/firearms_train_dc65943ea0be.jpg` | KleinsAd1963.jpg | Klein's Sporting Goods | Public domain | [Source](https://commons.wikimedia.org/wiki/File:KleinsAd1963.jpg) |
| `data/wikimedia_pilot/val/firearms/firearms_val_a1b84b3316e8.jpg` | Walther Arms, Inc Display at GAOS.jpg | Digitallymade | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Source](https://commons.wikimedia.org/wiki/File:Walther_Arms,_Inc_Display_at_GAOS.jpg) |
| `data/wikimedia_pilot/val/firearms/firearms_val_381b071aca85.jpg` | Heckler & Koch G3 Holzschaft Display noBG.png | Armémuseum (Swedish Army Museum) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Source](https://commons.wikimedia.org/wiki/File:Heckler_%26_Koch_G3_Holzschaft_Display_noBG.png) |
| `data/wikimedia_pilot/val/firearms/firearms_val_a7122c3120ca.jpg` | Heckler & Koch G3 Kunststoffschaft Display noBG.png | [Harald Hansen](https://commons.wikimedia.org/wiki/User:Harald_Hansen) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Source](https://commons.wikimedia.org/wiki/File:Heckler_%26_Koch_G3_Kunststoffschaft_Display_noBG.png) |
| `data/wikimedia_pilot/val/firearms/firearms_val_21db7f2e05ab.jpg` | RemingtonVestPocketNo1.jpg | DrReload | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Source](https://commons.wikimedia.org/wiki/File:RemingtonVestPocketNo1.jpg) |
| `data/wikimedia_pilot/test/firearms/firearms_test_7d806f2bc687.jpg` | Revolver Lefaucheux IMG 3108.jpg | Rama | [CC BY-SA 2.0 France](https://creativecommons.org/licenses/by-sa/2.0/fr/) | [Source](https://commons.wikimedia.org/wiki/File:Revolver_Lefaucheux_IMG_3108.jpg) |
| `data/wikimedia_pilot/test/firearms/firearms_test_b97b4eccd9c2.jpg` | Revolver mod 1878 IMG 3100.jpg | Rama | [CC BY-SA 2.0 France](https://creativecommons.org/licenses/by-sa/2.0/fr/) | [Source](https://commons.wikimedia.org/wiki/File:Revolver_mod_1878_IMG_3100.jpg) |
| `data/wikimedia_pilot/test/firearms/firearms_test_8f545103cb00.jpg` | .500 Linebaugh revolver variant.jpg | SEMMERLING | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [Source](https://commons.wikimedia.org/wiki/File:.500_Linebaugh_revolver_variant.jpg) |
| `data/wikimedia_pilot/train/explosives/explosives_train_15618430cce2.jpg` | Gyunyusekken Noren Summer.jpg | ITA-ATU, photographer; Cow Brand Soap Kyoshinsha Co., Ltd., banner maker | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [Source](https://commons.wikimedia.org/wiki/File:Gyunyusekken_Noren_Summer.jpg) |
| `data/wikimedia_pilot/train/explosives/explosives_train_73cb665d7a67.jpg` | Japan Type 10 grenade discharger.jpg | No machine-readable author; Makthorpe assumed from the recorded copyright claim | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/) | [Source](https://commons.wikimedia.org/wiki/File:Japan_Type_10_grenade_discharger.jpg) |
| `data/wikimedia_pilot/val/explosives/explosives_val_dd020909c102.jpg` | 1918 German UXOs2.JPG | John Warwick Brooke | Public domain | [Source](https://commons.wikimedia.org/wiki/File:1918_German_UXOs2.JPG) |
| `data/wikimedia_pilot/val/explosives/explosives_val_1599c9e8a92a.jpg` | Aircraft rocket and explosive ordnance at Swiss Air Force Museum, Dubendorf (Ank Kumar) 06.jpg | Ank Kumar | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Source](https://commons.wikimedia.org/wiki/File:Aircraft_rocket_and_explosive_ordnance_at_Swiss_Air_Force_Museum,_Dubendorf_%28Ank_Kumar%29_06.jpg) |
| `data/wikimedia_pilot/test/explosives/explosives_test_f3d73765d23b.jpg` | ACFD-Bomb-Disposal-Unit.jpg | Zackmann08 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [Source](https://commons.wikimedia.org/wiki/File:ACFD-Bomb-Disposal-Unit.jpg) |
| `data/wikimedia_pilot/train/financial_promotion/financial_promotion_train_cf68dfd0fea9.jpg` | Los Angeles-Bullfrog Realty & Investment Co. 1905 advertisement.png | Los Angeles-Bullfrog Realty & Investment Co. | Public domain | [Source](https://commons.wikimedia.org/wiki/File:Los_Angeles-Bullfrog_Realty_%26_Investment_Co._1905_advertisement.png) |
| `data/wikimedia_pilot/train/financial_promotion/financial_promotion_train_1aed323c38cb.jpg` | What Safer Investment Than Your Own Business? - DPLA - e9e54ff809c9b4495b8d17fec1e0b485.jpg | Dow Chemical Company; source record supplied by Science History Institute through DPLA | Public domain | [Source](https://commons.wikimedia.org/wiki/File:What_Safer_Investment_Than_Your_Own_Business%3F_-_DPLA_-_e9e54ff809c9b4495b8d17fec1e0b485.jpg) |
| `data/wikimedia_pilot/test/financial_promotion/financial_promotion_test_128923d1c757.jpg` | Bitsquare.png | Calva82 | [CC BY-SA 4.0, historical registry record](https://creativecommons.org/licenses/by-sa/4.0/) | [Deleted source page](https://commons.wikimedia.org/wiki/File:Bitsquare.png) |

## Deleted `Bitsquare.png` source

Wikimedia Commons deleted `File:Bitsquare.png` at 2026-08-26 11:06:56 UTC following a discussion that classified the upload as promotional spam and out of scope. The deletion discussion did not expressly identify a copyright violation. The local registry predates deletion and records:

- creator: `Calva82`;
- terms: `CC BY-SA 4.0`;
- source SHA-1: `128923d1c7575e9d456f6bc710d295462dca8606`;
- downloaded SHA-256: `c836b4e82aca68db8f77063764fb4d36ea6f3de58875109efb43a2012e5e5f22`; and
- normalized local SHA-256: `0d567cbb6a184d8d07ab8898fc7f828da78fed6c73983111c95831c2c0489a2f`.

Official deletion discussion: https://commons.wikimedia.org/wiki/Commons:Deletion_requests/Files_uploaded_by_Calva82

The file remains relevant to the saved historical diagnostic, but its source is no longer independently downloadable. Preserve the registry and this caveat with any retained copy. Prefer a newly reviewed, live-source replacement in a future diagnostic version, and do not silently substitute that replacement into the saved evaluation run.

## Excluded title-search collision

`data/wikimedia_pilot/train/explosives/explosives_train_7c750c3977f9.jpg` is not one of the 26 retained diagnostic items. Manual review found a bridge in Grenade, Spain, with no visible explosive content. The file entered the candidate set because of a title-search collision. If it is retained for audit history, it remains subject to its recorded [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) terms and [Commons source page](https://commons.wikimedia.org/wiki/File:Grenade_-_pont_sur_la_save.jpg).

## Trademark and personality-rights note

Current Commons metadata marks `Gyunyusekken Noren Summer.jpg` as trademarked. Its Creative Commons license covers applicable copyright rights, not trademark use. No file may be presented as endorsed by a depicted brand, creator, museum, organization, or person.

Images containing identifiable people or brands can also involve privacy, publicity, personality, moral, or trademark rights that are independent of copyright. The registry does not contain model releases. Public or commercial reusers must conduct any additional review required in their jurisdiction.

Sources:

- Wikimedia Commons reuse guidance: https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia
- Wikimedia Commons Imageinfo API: https://www.mediawiki.org/wiki/API:Imageinfo
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
- CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
- CC BY-SA 2.5: https://creativecommons.org/licenses/by-sa/2.5/
- CC BY-SA 2.0 France: https://creativecommons.org/licenses/by-sa/2.0/fr/
