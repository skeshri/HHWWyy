import json

input_var_jsonFile = open('input_variables.json','r')
input_var_jsonFile1 = open('allVar.json','r')

variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()
variable_list1 = json.load(input_var_jsonFile1,encoding="utf-8").items()

# print(variable_list)
# print(variable_list["UsedVariables"])

print(variable_list)
# for key,var in variable_list:
#     print key,var

print("-"*51)
print(variable_list1[0])
print("-"*51)
# for key,var in variable_list1["Comment"]:
# for key,var in variable_list1[0]:
    # print key,var
#
print (variable_list1[0][1][0])
for key,var in variable_list1[0][1][0]:
    print key,var
# for item in variable_list1[0]:
    # print item
    # print "===="
    # for this in item:
        # print this
        # for key, var in this:
            # print ("==> ",key,var)
    # print "++++"
