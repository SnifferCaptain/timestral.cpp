# 从原核生成预算变体，不修改原GLSL、绑定或push布局
file(READ "${SOURCE}" TEXT)
string(FIND "${TEXT}" "constant_id" EXISTING_SPECIALIZATION)
if(NOT EXISTING_SPECIALIZATION EQUAL -1)
    message(FATAL_ERROR "预算生成器要求先审查已有specialization常量")
endif()
string(REGEX MATCH "layout[ \t]*[(]" FIRST_LAYOUT "${TEXT}")
if(NOT FIRST_LAYOUT)
    message(FATAL_ERROR "预算shader没有layout声明")
endif()
string(FIND "${TEXT}" "${FIRST_LAYOUT}" POSITION)
string(SUBSTRING "${TEXT}" 0 ${POSITION} PREFIX)
string(SUBSTRING "${TEXT}" ${POSITION} -1 BODY)
set(TEXT "${PREFIX}layout(constant_id = 255) const uint budgetGridWidth = 0;

${BODY}")
string(REPLACE "gl_NumWorkGroups.x" "budgetGridWidth" TEXT "${TEXT}")
string(REPLACE "void main() {" "void main() {
    if (budgetGridWidth == 0) {
        return;
    }" TEXT "${TEXT}")
file(WRITE "${OUTPUT}" "${TEXT}")
