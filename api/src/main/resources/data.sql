-- 1. 광역시들(insert ignore로 중복 방지)
INSERT IGNORE INTO region (region_name)
VALUES ('강원특별자치도'),
       ('경기도'),
       ('경상남도'),
       ('경상북도'),
       ('광주광역시'),
       ('대구광역시'),
       ('대전광역시'),
       ('부산광역시'),
       ('서울특별시'),
       ('세종특별자치시'),
       ('울산광역시'),
       ('인천광역시'),
       ('전라남도'),
       ('전북특별자치도'),
       ('제주특별자치도'),
       ('충청남도'),
       ('충청북도');

-- 2. 대전광역시 하위 구(區)도 INSERT IGNORE 로 중복 방지
INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '유성구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '대덕구', region_id
FROM region
WHERE region_name = '대전광역시';