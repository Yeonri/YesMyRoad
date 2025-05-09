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

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '군포시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '김포시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남양주시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동두천시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '부천시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '성남시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '수원시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '시흥시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '안산시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '안성시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '안양시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '양주시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '양평군', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '여주시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '연천군', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '오산시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '용인시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '의왕시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '의정부시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '이천시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '파주시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '평택시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '포천시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '하남시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '화성시', region_id
FROM region
WHERE region_name = '경기도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '거제시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '거창군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '고성군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '김해시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남해군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '밀양시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '사천시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '산청군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '양산시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '의령군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '진주시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '창녕군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '창원시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '통영시', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '하동군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '함안군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '함양군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '합천군', region_id
FROM region
WHERE region_name = '경상남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '경산시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '경주시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '고령군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '구미시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '김천시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '문경시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '봉화군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '상주시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '성주군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '안동시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영덕군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영양군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영주시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영천시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '예천군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '울릉군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '울진군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '의성군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '청도군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '청송군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '칠곡군', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '포항시', region_id
FROM region
WHERE region_name = '경상북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '광산구', region_id
FROM region
WHERE region_name = '광주광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남구', region_id
FROM region
WHERE region_name = '광주광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '광주광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '북구', region_id
FROM region
WHERE region_name = '광주광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서구', region_id
FROM region
WHERE region_name = '광주광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '군위군', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '달서구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '달성군', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '북구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '수성구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '대구광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '대덕구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
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
SELECT '중구', region_id
FROM region
WHERE region_name = '대전광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강서구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '금정구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '기장군', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동래구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '부산진구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '북구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '사상구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '사하구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '수영구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '연제구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영도구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '해운대구', region_id
FROM region
WHERE region_name = '부산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강남구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강동구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강북구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강서구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '관악구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '광진구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '구로구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '금천구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '노원구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '도봉구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동대문구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동작구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '마포구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서대문구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서초구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '성동구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '성북구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '송파구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '양천구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영등포구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '용산구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '은평구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '종로구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중랑구', region_id
FROM region
WHERE region_name = '서울특별시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '', region_id
FROM region
WHERE region_name = '세종특별자치시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남구', region_id
FROM region
WHERE region_name = '울산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '울산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '북구', region_id
FROM region
WHERE region_name = '울산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '울주군', region_id
FROM region
WHERE region_name = '울산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '울산광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강화군', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '계양구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남동구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '동구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '미추홀구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '부평구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '연수구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '옹진군', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '중구', region_id
FROM region
WHERE region_name = '인천광역시';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '강진군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '고흥군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '곡성군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '광양시', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '구례군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '나주시', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '담양군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '목포시', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '무안군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '보성군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '순천시', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '신안군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '여수시', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영광군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영암군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '완도군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '장성군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '장흥군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '진도군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '함평군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '해남군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '화순군', region_id
FROM region
WHERE region_name = '전라남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '고창군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '군산시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '김제시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '남원시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '무주군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '부안군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '순창군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '완주군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '익산시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '임실군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '장수군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '전주시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '정읍시', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '진안군', region_id
FROM region
WHERE region_name = '전북특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서귀포시', region_id
FROM region
WHERE region_name = '제주특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '제주시', region_id
FROM region
WHERE region_name = '제주특별자치도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '계룡시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '공주시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '금산군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '논산시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '당진시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '보령시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '부여군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서산시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '서천군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '아산시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '예산군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '천안시', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '청양군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '태안군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '홍성군', region_id
FROM region
WHERE region_name = '충청남도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '괴산군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '단양군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '보은군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '영동군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '옥천군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '음성군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '제천시', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '증평군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '진천군', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '청주시', region_id
FROM region
WHERE region_name = '충청북도';

INSERT IGNORE INTO region (region_name, parent_region_id)
SELECT '충주시', region_id
FROM region
WHERE region_name = '충청북도';